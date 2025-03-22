import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .pointnet_stack_utils import *
from .model_utils import *
from scipy.optimize import linear_sum_assignment
from utils import loss_utils
import pc_util
import torch_scatter

class ClusterRefineNet(nn.Module):
    def __init__(self, model_cfg, input_channel):
        super().__init__()
        self.model_cfg = model_cfg
        self.matcher = HungarianMatcher(self.model_cfg.MatchRadius)
        sa_cfg = model_cfg.RefineSA
        mlps = sa_cfg.MLPs
        mlps = [[input_channel] + mlp for mlp in mlps]
        self.fea_refine_module = StackSAModuleMSG(
            radii=sa_cfg.Radii,
            nsamples=sa_cfg.Nsamples,
            mlps=mlps,
            use_xyz=True,
            pool_method='max_pool'
        )
        self.num_output_feature = sum([mlp[-1] for mlp in mlps])
        self.shared_fc = LinearBN(256, 128)
        self.drop = nn.Dropout(0.5)
        self.offset_fc = nn.Linear(128, 3)
        
        if self.training:
            self.train_dict = {}
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss()
            )
            self.loss_weight = self.model_cfg.LossWeight
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def supervised_kmeans_cluster(self, pts, features, score_thresh, n_clusters_max=50):
        """
        使用监督约束的 K-Means 聚类
        Args:
            pts: [B*N, 3] 候选拐点坐标
            features: [B*N, C] 点特征
            score_thresh: 分数阈值
            n_clusters_max: 最大簇数（防止过分割）
        Returns:
            key_pts: [M, 3] 聚类中心
            num_cluster: 簇数
        """
        # 筛选高置信度点
        mask = pts[:, 0] > -5  # 假设未筛选点坐标为 -10
        if not mask.any():
            return pts.new_zeros(0, 3), 0
        
        valid_pts = pts[mask]  # [N_valid, 3]
        valid_features = features[mask]  # [N_valid, C]
        
        # 估计簇数：基于点密度和特征相似性
        dist_matrix = torch.cdist(valid_pts, valid_pts)  # [N_valid, N_valid]
        min_dist = dist_matrix.topk(k=2, largest=False, dim=1)[0][:, 1]  # 每个点到最近邻的距离
        avg_min_dist = min_dist.mean().item()
        n_clusters_est = min(int(valid_pts.size(0) * avg_min_dist / self.model_cfg.MatchRadius), n_clusters_max)
        n_clusters = max(1, min(n_clusters_est, valid_pts.size(0) // 2))  # 至少 1 个簇
        
        # 特征增强聚类
        cluster_input = torch.cat([valid_pts, valid_features * 0.1], dim=-1)  # [N_valid, 3+C]，特征缩放避免主导
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(cluster_input.cpu().numpy())
        labels = torch.from_numpy(labels).to(pts.device, dtype=torch.int64)  # 转换为 int64
        
        # 计算聚类中心
        key_pts = torch_scatter.scatter_mean(valid_pts, labels, dim=0)
        num_cluster = key_pts.size(0)
        
        # 过滤无效簇（若有）
        valid_mask = torch.sum(key_pts, dim=-1) > -1e-5
        key_pts = key_pts[valid_mask]
        num_cluster = key_pts.size(0)
        
        return key_pts, num_cluster

    def forward(self, batch_dict):
        offset_pts = batch_dict['points'].clone()
        offset_pts = offset_pts[:, :, :3]
        offset = batch_dict['point_pred_offset']
        pts_score = batch_dict['point_pred_score']
        score_thresh = self.model_cfg.ScoreThresh
        
        # 调整候选拐点
        offset_pts[pts_score > score_thresh] += offset[pts_score > score_thresh]
        pts_cluster = offset_pts.new_ones(offset_pts.shape) * -10
        pts_cluster[pts_score > score_thresh] = offset_pts[pts_score > score_thresh]
        
        # 使用监督 K-Means 替换 DBSCAN
        batch_size = offset_pts.size(0)
        key_pts_list, num_cluster_list = [], []
        for i in range(batch_size):
            pts = pts_cluster[i]  # [N, 3]
            features = batch_dict['point_features'][i]  # [N, C]
            key_pts, num_cluster = self.supervised_kmeans_cluster(pts, features, score_thresh)
            key_pts_list.append(key_pts)
            num_cluster_list.append(num_cluster)
        
        # 处理空簇情况
        if sum(num_cluster_list) == 0:
            batch_dict['warning'] = True
            return batch_dict
        
        key_pts = torch.cat(key_pts_list, dim=0)  # [M, 3]
        
        
        # 保存 offset_pts 和 key_pts 到文件
        # save_dir = '/data/haoran/Point2Roof/cluster_refine_kmeans_res'
        # os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）
        # from datetime import datetime
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 添加时间戳
        
        # for b in range(offset_pts.shape[0]):
        #     filename = os.path.join(save_dir, f'batch_{b}_{timestamp}.xyz')
        #     pts = offset_pts[b].detach().cpu().numpy()  # [N, 3]
        #     scores = pts_score[b].detach().cpu().numpy()  # [N]
        #     batch_key_pts = key_pts_list[b].detach().cpu().numpy()  # [M_b, 3]
            
        #     with open(filename, 'w') as f:
        #         # 保存 offset_pts
        #         for i, pt in enumerate(pts):
        #             if scores[i] > score_thresh:
        #                 color = "255 0 0"  # 红色
        #             else:
        #                 color = "255 255 255"  # 白色
        #             f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {color}\n")
                
        #         # 保存 key_pts，设置为黄色
        #         for kp in batch_key_pts:
        #             color = "255 255 0"  # 黄色
        #             f.write(f"{kp[0]:.6f} {kp[1]:.6f} {kp[2]:.6f} {color}\n")
        
        if self.training:
            new_pts, targets, labels, matches, new_xyz_batch_cnt = self.matcher(key_pts, batch_dict['vectors'])
            offset_targets = (targets - new_pts) / self.model_cfg.MatchRadius if new_pts is not None else None
            batch_dict['matches'] = matches
            self.train_dict.update({
                'keypoint_cls_label': labels,
                'keypoint_offset_label': offset_targets
            })
        else:
            pts_list, new_xyz_batch_cnt = [], []
            idx = 0
            for i, num in enumerate(num_cluster_list):
                if num == 0:
                    new_xyz_batch_cnt.append(0)
                    continue
                pts = key_pts[idx:idx + num]
                new_xyz_batch_cnt.append(num)
                pts_list.append(pts)
                idx += num
            if sum(new_xyz_batch_cnt) == 0:
                new_pts, new_xyz_batch_cnt = None, None
            else:
                new_pts = torch.cat(pts_list, 0)
                new_xyz_batch_cnt = new_pts.new_tensor(new_xyz_batch_cnt, dtype=torch.int32)

        if new_pts is None:
            print("new_pts is None")
            batch_dict['warning'] = True
            return batch_dict
        
        # 构造批次索引
        batch_idx = torch.zeros(new_pts.shape[0], device=new_pts.device)
        idx = 0
        for i, cnt in enumerate(new_xyz_batch_cnt):
            if cnt == 0:
                continue
            batch_idx[idx:idx + cnt] = i
            idx += cnt

        pos_mask = new_xyz_batch_cnt > 0
        offset_pts = offset_pts[pos_mask]
        xyz = offset_pts.view(-1, 3)
        xyz_batch_cnt = offset_pts.new_ones(offset_pts.shape[0], dtype=torch.int32) * offset_pts.shape[1]
        new_xyz_batch_cnt = new_xyz_batch_cnt[pos_mask]
        point_fea = batch_dict['point_features']
        point_fea = point_fea * pts_score.detach().unsqueeze(-1)
        point_fea = point_fea[pos_mask]
        point_fea = point_fea.contiguous().view(-1, point_fea.shape[-1])
        
        # 特征精炼
        _, refine_fea = self.fea_refine_module(xyz, xyz_batch_cnt, new_pts, new_xyz_batch_cnt, point_fea)

        x = self.drop(self.shared_fc(refine_fea))
        pred_offset = self.offset_fc(x)
        
        if self.training:
            self.train_dict.update({
                'keypoint_offset_pred': pred_offset
            })
        
        batch_dict['keypoint'] = torch.cat([batch_idx.view(-1, 1), new_pts], -1)
        batch_dict['keypoint_features'] = refine_fea
        batch_dict['refined_keypoint'] = pred_offset * self.model_cfg.MatchRadius + new_pts
        batch_dict['warning'] = False
        return batch_dict

    def loss(self, loss_dict, disp_dict):
        pred_offset = self.train_dict['keypoint_offset_pred']
        label_cls, label_offset = self.train_dict['keypoint_cls_label'], self.train_dict['keypoint_offset_label']
        reg_loss = self.get_reg_loss(pred_offset, label_offset, label_cls, self.loss_weight['reg_weight'])
        loss = reg_loss
        loss_dict.update({
            'refine_offset_loss': reg_loss.item(),
            'refine_loss': loss.item()
        })
        return loss, loss_dict, disp_dict

    def get_reg_loss(self, pred, label, cls_label, weight):
        positives = cls_label > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        reg_loss_src = self.reg_loss_func(pred.unsqueeze(dim=0), label.unsqueeze(dim=0), weights=reg_weights.unsqueeze(dim=0))
        reg_loss = reg_loss_src.sum()
        reg_loss = reg_loss * weight
        return reg_loss
    
    
class StackSAModuleMSG(nn.Module):

    def __init__(self, radii, nsamples, mlps, use_xyz, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](new_features)  # (1, C, M1 + M2 ..., nsample)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features


class HungarianMatcher(nn.Module):
    def __init__(self, match_r):
        super().__init__()
        self.dist_thresh = match_r

    # tips: matcher with dist threshold
    @torch.no_grad()
    def forward(self, output, targets):
        pts_list, target_list, label_list, match_list, new_xyz_batch_cnt = [], [], [], [], []
        for i in range(output.shape[0]):
            tmp_output, tmp_targets = output[i], targets[i]
            tmp_output = tmp_output[torch.sum(tmp_output, -1) > -2e1]
            if len(tmp_output) == 0:
                new_xyz_batch_cnt.append(0)
                continue
            tmp_targets = tmp_targets[torch.sum(tmp_targets, -1) > -2e1]
            vec_a = torch.sum(tmp_output.unsqueeze(1).repeat(1, tmp_targets.shape[0], 1) ** 2, -1)
            vec_b = torch.sum(tmp_targets.unsqueeze(0).repeat(tmp_output.shape[0], 1, 1) ** 2, -1)
            dist_matrix = vec_a + vec_b - 2 * torch.mm(tmp_output, tmp_targets.permute(1, 0))
            dist_matrix = F.relu(dist_matrix)
            dist_matrix = torch.sqrt(dist_matrix)

            out_ind, tar_ind = linear_sum_assignment(dist_matrix.cpu().numpy())
            out_ind, tar_ind = dist_matrix.new_tensor(out_ind, dtype=torch.int64), dist_matrix.new_tensor(tar_ind, dtype=torch.int64)
            dist_val = dist_matrix[out_ind, tar_ind]
            out_ind = out_ind[dist_val < self.dist_thresh]
            tar_ind = tar_ind[dist_val < self.dist_thresh]

            pts_list.append(tmp_output)
            tmp_label = tmp_targets.new_zeros(tmp_output.shape[0])
            tmp_label[out_ind] = 1.
            tmp_pts_target = tmp_targets.new_zeros(tmp_output.shape)
            tmp_pts_target[out_ind] = tmp_targets[tar_ind]
            tmp_match = tmp_targets.new_ones(tmp_output.shape[0], dtype=torch.int64) * -1
            tmp_match[out_ind] = tar_ind
            label_list.append(tmp_label)
            target_list.append(tmp_pts_target)
            match_list.append(tmp_match)
            new_xyz_batch_cnt.append(tmp_output.shape[0])
        if sum(new_xyz_batch_cnt) == 0:
            return None, None, None, None, None
        return torch.cat(pts_list, 0), torch.cat(target_list, 0), torch.cat(label_list, 0), torch.cat(match_list, 0), tmp_output.new_tensor(new_xyz_batch_cnt, dtype=torch.int32)
