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
    def __init__(self, model_cfg, input_channel=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.freeze = False
        self.input_channel = input_channel

        # 初始化模块为 None，等待 forward 中动态创建
        self.edge_feature_extractor = None
        self.global_att = None
        self.att_layer = None
        self.multi_scale_fusion = None
        self.shared_fc = None
        self.drop = nn.Dropout(0.5)
        self.cls_fc = None

        if self.training:
            self.train_dict = {}
            self.add_module('cls_loss_func', loss_utils.SigmoidBCELoss())
            self.loss_weight = self.model_cfg.LossWeight
            self.connectivity_weight = 0.2

        self.initialized = False  # 标记是否已初始化

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def initialize_modules(self, input_channel, device):
        """动态初始化所有模块并移动到指定设备"""
        self.input_channel = input_channel
        self.edge_feature_extractor = MultiScaleEdgeFeatureExtractor(self.input_channel).to(device)
        self.global_att = EfficientGlobalAttention(self.input_channel).to(device)
        self.att_layer = PairedPointAttention(self.input_channel).to(device)
        num_feature = self.att_layer.num_output_feature  # 动态获取，应为 input_channel
        self.multi_scale_fusion = HierarchicalFusion(self.input_channel).to(device)
        self.shared_fc = LinearBN(num_feature, num_feature).to(device)
        self.cls_fc = nn.Linear(num_feature, 1).to(device)
        self.init_weights()
        self.initialized = True

    def forward(self, batch_dict):
        batch_idx = batch_dict['keypoint'][:, 0]
        point_fea = batch_dict['keypoint_features']

        # 动态初始化模块
        if not self.initialized:
            if self.input_channel is None:
                self.input_channel = point_fea.shape[-1]  # 从 point_fea 推导，例如 256
            self.initialize_modules(self.input_channel, point_fea.device)

        if self.training:
            self.train_dict.clear()
            matches = batch_dict['matches']
            edge_label = batch_dict['edges']
            bin_label_list = []
            with torch.no_grad():
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
        with torch.no_grad():
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

        edge_fea = self.edge_feature_extractor(pair_fea1, pair_fea2)
        global_fea = self.global_att(edge_fea)
        fused_fea = self.multi_scale_fusion(edge_fea + global_fea)
        att_fea = self.att_layer(pair_fea1, pair_fea2)
        edge_fea = fused_fea + att_fea

        # 修复维度，确保 edge_fea 为 [N, C]
        if edge_fea.dim() == 3 and edge_fea.shape[0] == 1:
            edge_fea = edge_fea.squeeze(0)  # 从 [1, N, C] 调整为 [N, C]

        edge_pred = self.cls_fc(self.drop(self.shared_fc(edge_fea)))
        batch_dict['pair_points'] = torch.cat(pair_idx_list, 0)
        batch_dict['edge_score'] = torch.sigmoid(edge_pred).view(-1)

        if self.training:
            self.train_dict['edge_pred'] = edge_pred
            self.train_dict['pair_idx1'] = pair_idx1
            self.train_dict['pair_idx2'] = pair_idx2

        del edge_fea, global_fea, fused_fea, att_fea
        return batch_dict

    def loss(self, loss_dict, disp_dict):
        pred_cls = self.train_dict['edge_pred']
        label_cls = self.train_dict['label']

        cls_loss = self.get_cls_loss(pred_cls, label_cls, self.loss_weight['cls_weight'])
        pair_idx1 = self.train_dict['pair_idx1']
        pair_idx2 = self.train_dict['pair_idx2']
        connectivity_loss = self.get_connectivity_loss(pred_cls, pair_idx1, pair_idx2)
        loss = cls_loss + self.connectivity_weight * connectivity_loss

        loss_dict.update({
            'edge_cls_loss': cls_loss.item(),
            'edge_connectivity_loss': connectivity_loss.item(),
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

    def get_cls_loss(self, pred, label, weight):
        positives = label > 0
        negatives = label == 0
        cls_weights = (negatives * 1.0 + positives * 1.0).float()
        pos_normalizer = positives.sum().float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss_src = self.cls_loss_func(pred.squeeze(-1), label, weights=cls_weights)
        cls_loss = cls_loss_src.sum()
        return cls_loss * weight

    def get_connectivity_loss(self, pred, pair_idx1, pair_idx2):
        pred_sigmoid = torch.sigmoid(pred.squeeze(-1))
        pred_positive = pred_sigmoid >= 0.5
        num_pred_positive = pred_positive.sum().float()

        if num_pred_positive == 0:
            return torch.tensor(0.0, device=pred.device)

        connectivity_loss = 0.0
        for i in range(pred_sigmoid.shape[0]):
            if pred_positive[i]:
                mask1 = (pair_idx1 == pair_idx1[i]) | (pair_idx2 == pair_idx1[i])
                mask2 = (pair_idx1 == pair_idx2[i]) | (pair_idx2 == pair_idx2[i])
                neighbors = (mask1 | mask2) & pred_positive
                num_neighbors = neighbors.sum().float() - 1
                if num_neighbors <= 0:
                    connectivity_loss += (1 - pred_sigmoid[i]) ** 2

        return connectivity_loss / max(num_pred_positive, 1)

class MultiScaleEdgeFeatureExtractor(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channel * 2, input_channel, 1)
        self.conv2 = nn.Conv1d(input_channel * 2, input_channel, 3, padding=1, groups=4)
        self.conv3 = nn.Conv1d(input_channel * 2, input_channel, 5, padding=2, groups=4)
        self.fusion = nn.Conv1d(input_channel * 3, input_channel, 1)

    def forward(self, fea1, fea2):
        edge_fea = torch.cat([fea1, fea2], dim=-1).unsqueeze(0).permute(0, 2, 1)
        fea_small = F.relu(self.conv1(edge_fea))
        fea_mid = F.relu(self.conv2(edge_fea))
        fea_large = F.relu(self.conv3(edge_fea))
        fused_fea = self.fusion(torch.cat([fea_small, fea_mid, fea_large], dim=1))
        return fused_fea.transpose(1, 2)

class EfficientGlobalAttention(nn.Module):
    def __init__(self, input_channel, num_clusters=8, sample_ratio=0.25):
        super().__init__()
        self.input_channel = input_channel
        self.num_clusters = num_clusters  # 动态分组数
        self.sample_ratio = sample_ratio  # 稀疏采样比例
        self.proj = nn.Linear(input_channel, input_channel // 4)
        self.reproj = nn.Linear(input_channel // 4, input_channel)
        self.norm = nn.LayerNorm(input_channel)
        self.cluster_proj = nn.Linear(input_channel, num_clusters)  # 用于动态分组

    def forward(self, x):
        if x.dim() == 3:
            batch_size, N, C = x.shape
            x_flat = x.view(batch_size * N, C)
        else:
            x_flat = x

        # 动态分组：通过投影计算每个token属于哪个簇的概率
        cluster_scores = F.softmax(self.cluster_proj(x_flat), dim=-1)  # (batch_size * N, num_clusters)
        _, cluster_indices = cluster_scores.max(dim=-1)  # (batch_size * N)

        # 稀疏采样：每个簇内选择部分代表性token
        sampled_indices = []
        for c in range(self.num_clusters):
            cluster_mask = (cluster_indices == c)
            cluster_size = cluster_mask.sum().item()
            if cluster_size > 0:
                num_samples = max(1, int(cluster_size * self.sample_ratio))
                cluster_idx = torch.where(cluster_mask)[0]
                sampled_idx = cluster_idx[torch.randperm(cluster_size)[:num_samples]]
                sampled_indices.append(sampled_idx)
        sampled_indices = torch.cat(sampled_indices)  # (sampled_N)

        # 对采样点计算注意力
        x_sampled = x_flat[sampled_indices]  # (sampled_N, C)
        q = self.proj(x_sampled)  # (sampled_N, C//4)
        k = self.proj(x_sampled)  # (sampled_N, C//4)
        v = self.proj(x_sampled)  # (sampled_N, C//4)

        # 注意力计算，复杂度从 O(N²) 降到 O(sampled_N²)，sampled_N << N
        attn = F.softmax(q @ k.T / (q.shape[-1] ** 0.5), dim=-1)  # (sampled_N, sampled_N)
        out_sampled = attn @ v  # (sampled_N, C//4)

        # 将采样结果插值回完整序列
        out = torch.zeros_like(x_flat[:, :C//4])  # (batch_size * N, C//4)
        out[sampled_indices] = out_sampled
        out = self.reproj(out)  # (batch_size * N, C)

        # 残差连接和归一化
        out = self.norm(x_flat + out)

        if x.dim() == 3:
            return out.view(batch_size, N, C)
        return out

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

class PairedPointAttention(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.edge_att1 = nn.Sequential(
            nn.Linear(input_channel, input_channel),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(),
            nn.Linear(input_channel, input_channel),
            nn.Sigmoid(),
        )
        self.edge_att2 = nn.Sequential(
            nn.Linear(input_channel, input_channel),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(),
            nn.Linear(input_channel, input_channel),
            nn.Sigmoid(),
        )
        self.fea_fusion_layer = nn.MaxPool1d(2)
        self.num_output_feature = input_channel

    def forward(self, point_fea1, point_fea2):
        fusion_fea = point_fea1 + point_fea2
        att1 = self.edge_att1(fusion_fea)
        att2 = self.edge_att2(fusion_fea)
        att_fea1 = point_fea1 * att1
        att_fea2 = point_fea2 * att2
        fea = torch.cat([att_fea1.unsqueeze(1), att_fea2.unsqueeze(1)], 1)
        fea = self.fea_fusion_layer(fea.permute(0, 2, 1)).squeeze(-1)
        return fea