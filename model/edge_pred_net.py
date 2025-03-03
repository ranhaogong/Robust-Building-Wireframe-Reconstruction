import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_stack_utils import *
from .model_utils import *
from scipy.optimize import linear_sum_assignment
from utils import loss_utils
import pc_util
import itertools

class EdgeAttentionNet(nn.Module):
    def __init__(self, model_cfg, input_channel):
        super().__init__()
        self.model_cfg = model_cfg
        self.freeze = False

        self.edge_conv = DynamicEdgeGraphConv(input_channel)
        self.transformer = ContextAwareTransformer(input_channel)
        self.att_layer = DualCrossAttention(input_channel)
        num_feature = self.att_layer.num_output_feature
        self.multi_scale_fusion = HierarchicalFusion(input_channel)

        self.shared_fc = LinearBN(num_feature, num_feature)
        self.drop = nn.Dropout(0.5)
        self.cls_fc = nn.Linear(num_feature, 1)

        if self.training:
            self.train_dict = {}
            self.add_module('cls_loss_func', loss_utils.SigmoidBCELoss())
            self.loss_weight = self.model_cfg.LossWeight
            self.alpha = 0.25
            self.gamma = 2.0

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        batch_idx = batch_dict['keypoint'][:, 0]
        point_fea = batch_dict['keypoint_features']

        if self.training:
            self.train_dict.clear()  # 每个 forward 清空 train_dict
            matches = batch_dict['matches']
            edge_label = batch_dict['edges']
            bin_label_list = []
            with torch.no_grad():  # 减少内存占用
                for i, edge in enumerate(edge_label):
                    mask = batch_idx == i
                    tmp_idx = batch_idx[mask]
                    if tmp_idx.shape[0] <= 1:
                        continue
                    match = matches[mask]
                    match_edge = list(itertools.combinations(match.cpu().numpy(), 2))
                    match_edge = [tuple(sorted(e)) for e in match_edge]
                    edge = [tuple(e) for e in edge.cpu().numpy()]
                    label = edge_label.new_tensor([e in edge for e in match_edge])
                    bin_label_list.append(label)
            self.train_dict['label'] = torch.cat(bin_label_list)

        idx = 0
        pair_idx_list, pair_idx_list1, pair_idx_list2 = [], [], []
        with torch.no_grad():  # 减少点对生成内存
            for i in range(batch_dict['batch_size']):
                mask = batch_idx == i
                tmp_idx = batch_idx[mask]
                if tmp_idx.shape[0] <= 1:
                    continue
                fea = point_fea[mask]
                pair_idx = itertools.combinations(range(fea.shape[0]), 2)
                pair_idx = point_fea.new_tensor(list(pair_idx))
                pair_idx_list.append(pair_idx)
                pair_idx_list1.append(pair_idx[:, 0] + idx)
                pair_idx_list2.append(pair_idx[:, 1] + idx)
                idx += tmp_idx.shape[0]

        pair_idx1 = torch.cat(pair_idx_list1).long()
        pair_idx2 = torch.cat(pair_idx_list2).long()
        pair_fea1 = point_fea[pair_idx1]
        pair_fea2 = point_fea[pair_idx2]

        edge_fea = self.edge_conv(pair_fea1, pair_fea2)
        global_fea = self.transformer(edge_fea)
        att_fea = self.att_layer(pair_fea1, pair_fea2)
        fused_fea = self.multi_scale_fusion(edge_fea + global_fea + att_fea)
        fused_fea = fused_fea.squeeze(0)
        edge_pred = self.cls_fc(self.drop(self.shared_fc(fused_fea)))
        batch_dict['pair_points'] = torch.cat(pair_idx_list, 0)
        batch_dict['edge_score'] = torch.sigmoid(edge_pred).view(-1)

        if self.training:
            self.train_dict['edge_pred'] = edge_pred

        # 显式清理中间变量
        del edge_fea, global_fea, att_fea, fused_fea
        return batch_dict

    def loss(self, loss_dict, disp_dict):
        pred_cls = self.train_dict['edge_pred']
        label_cls = self.train_dict['label']
        loss = self.get_focal_loss(pred_cls, label_cls, self.loss_weight['cls_weight'])

        loss_dict.update({
            'edge_cls_loss': loss.item(),
            'edge_loss': loss.item()
        })

        pred_cls = pred_cls.squeeze(-1)
        label_cls = label_cls.squeeze(-1)
        pred_logit = torch.sigmoid(pred_cls)
        pred = torch.where(pred_logit >= 0.5, pred_logit.new_ones(pred_logit.shape),
                           pred_logit.new_zeros(pred_logit.shape))
        acc = torch.sum((pred == label_cls) & (label_cls == 1)).item() / max(torch.sum(label_cls == 1).item(), 1)
        disp_dict.update({'edge_acc': acc})
        return loss, loss_dict, disp_dict

    def get_focal_loss(self, pred, label, weight):
        pred_sigmoid = torch.sigmoid(pred.squeeze(-1))
        label = label.float()
        pt = pred_sigmoid * label + (1 - pred_sigmoid) * (1 - label)
        alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label)
        focal_loss = -alpha_t * (1 - pt).pow(self.gamma) * torch.log(pt + 1e-8)
        return focal_loss.mean() * weight

class DynamicEdgeGraphConv(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channel * 2, input_channel, 1)
        self.conv2 = nn.Conv1d(input_channel, input_channel, 1)
        self.edge_weight = nn.Linear(input_channel * 2, 1)

    def forward(self, fea1, fea2):
        edge_fea = torch.cat([fea1, fea2], dim=-1)
        weights = torch.sigmoid(self.edge_weight(edge_fea))
        edge_fea = edge_fea.unsqueeze(0).permute(0, 2, 1)
        edge_fea = F.relu(self.conv1(edge_fea))
        weights = weights.view(1, 1, -1)
        edge_fea = edge_fea * weights
        edge_fea = F.relu(self.conv2(edge_fea))
        return edge_fea.transpose(1, 2)

class ContextAwareTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=256)
        self.context_gate = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context_weights = self.context_gate(x)
        x = self.transformer(x) * context_weights
        return x

class HierarchicalFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scale1 = nn.Linear(input_dim, input_dim // 2)
        self.scale2 = nn.Linear(input_dim, input_dim)
        self.up = nn.Linear(input_dim // 2, input_dim)
        self.final = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x):
        s1 = F.relu(self.scale1(x))
        s2 = F.relu(self.scale2(x))
        s1_up = F.relu(self.up(s1))
        fused = self.final(torch.cat([s1_up, s2], dim=-1))
        return fused + x

class DualCrossAttention(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.query1 = nn.Linear(input_channel, input_channel)
        self.key1 = nn.Linear(input_channel, input_channel)
        self.value1 = nn.Linear(input_channel, input_channel)
        self.query2 = nn.Linear(input_channel, input_channel)
        self.key2 = nn.Linear(input_channel, input_channel)
        self.value2 = nn.Linear(input_channel, input_channel)
        self.num_output_feature = input_channel
        self.chunk_size = 4096  # 分块大小，减少内存占用

    def forward(self, point_fea1, point_fea2):
        num_edges = point_fea1.shape[0]
        fea_out = torch.zeros_like(point_fea1)

        # 分块处理注意力计算
        for i in range(0, num_edges, self.chunk_size):
            end = min(i + self.chunk_size, num_edges)
            fea1_chunk = point_fea1[i:end]
            fea2_chunk = point_fea2[i:end]

            Q1 = self.query1(fea1_chunk)
            K1 = self.key1(point_fea2)  # 全量 K1，但只计算 chunk 部分
            V1 = self.value1(point_fea2)
            attn1 = F.softmax(Q1 @ K1.T / (Q1.shape[-1] ** 0.5), dim=-1)
            fea1 = attn1 @ V1

            Q2 = self.query2(fea2_chunk)
            K2 = self.key2(point_fea1)
            V2 = self.value2(point_fea1)
            attn2 = F.softmax(Q2 @ K2.T / (Q2.shape[-1] ** 0.5), dim=-1)
            fea2 = attn2 @ V2

            fea_out[i:end] = fea1 + fea2

        return fea_out