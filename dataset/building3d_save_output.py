import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import os
import shutil
import open3d as o3d

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from pathlib import Path
from tqdm import tqdm  # 导入 tqdm 库
import time

def read_ply(pts_file):
    # 使用 open3d 读取 ply 文件
    pcd = o3d.io.read_point_cloud(pts_file)
    
    # 提取点云中的点数据
    pts = np.asarray(pcd.points)
    # 返回前三列数据（即点云的坐标）
    # print(pts)
    return pts

def read_pts(pts_file, color=False, nir=False, intensity=False):
    with open(pts_file, 'r') as f:
        lines = f.readlines()
        data = [line.strip().split(' ') for line in lines]
        
    # 将数据转化为 NumPy 数组，方便后续处理
    data = np.array(data, dtype=np.float64)

    if color == False and nir == False and intensity == False:
        # 只读取前三列 (x, y, z)
        pts = data[:, :3]
        return pts

    if color == True and nir == False and intensity == False:
        # 读取前三列 (x, y, z) 和 RGB (r, g, b)
        pts = data[:, :6]  # x, y, z, r, g, b
        return pts

    if color == False and nir == True and intensity == False:
        # 读取前三列 (x, y, z) 和 NIR
        pts = data[:, [0, 1, 2, 6]]  # x, y, z, nir
        return pts

    if color == False and nir == False and intensity == True:
        # 读取前三列 (x, y, z) 和 Intensity
        pts = data[:, [0, 1, 2, 7]]  # x, y, z, intensity
        return pts

    if color == True and nir == True and intensity == False:
        # 读取前三列 (x, y, z), RGB 和 NIR
        pts = data[:, [0, 1, 2, 3, 4, 5, 6]]  # x, y, z, r, g, b, nir
        return pts

    if color == True and nir == False and intensity == True:
        # 读取前三列 (x, y, z), RGB 和 Intensity
        pts = data[:, [0, 1, 2, 3, 4, 5, 7]]  # x, y, z, r, g, b, intensity
        return pts

    if color == False and nir == True and intensity == True:
        # 读取前三列 (x, y, z), NIR 和 Intensity
        pts = data[:, [0, 1, 2, 6, 7]]  # x, y, z, nir, intensity
        return pts

    if color == True and nir == True and intensity == True:
        # 读取前三列 (x, y, z), RGB, NIR 和 Intensity
        pts = data[:, [0, 1, 2, 3, 4, 5, 6, 7]]  # x, y, z, r, g, b, nir, intensity
        return pts

def load_obj(obj_file):
    vs, edges = [], set()
    with open(obj_file, 'r') as f:
        lines = f.readlines()
    for f in lines:
        vals = f.strip().split(' ')
        if vals[0] == 'v':
            vs.append(vals[1:])
        elif vals[0] == 'l':
            e = [int(vals[1]) - 1, int(vals[2]) - 1]
            edges.add(tuple(sorted(e)))
    vs = np.array(vs, dtype=np.float64)
    edges = np.array(list(edges))
    return vs, edges

def writePoints(points, clsRoad):
    with open(clsRoad, 'w+') as file1:
        for i in range(len(points)):
            point = points[i]
            file1.write(str(point[0]))
            file1.write(' ')
            file1.write(str(point[1]))
            file1.write(' ')
            file1.write(str(point[2]))
            file1.write(' ')
            file1.write('\n')


class Building3DDatasetOutput(Dataset):
    def __init__(self, data_path, transform, data_cfg, logger=None, color=False, nir=False, intensity=False, fpfh=False):
        with open(data_path, 'r') as f:
            self.file_list = f.readlines()
        self.file_list = [f.strip() for f in self.file_list]
        flist = []
        for l in self.file_list:
             flist.append(l)
        self.file_list = flist

        self.npoint = data_cfg.NPOINT

        self.transform = transform

        self.color = color
        
        self.nir = nir
        
        self.intensity = intensity

        self.fpfh = fpfh
        
        if logger is not None:
            logger.info('Total samples: %d' % len(self))

    def __len__(self):
        return len(self.file_list)
    def compute_fpfh_features(self, points):
        # Convert points to open3d format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # assuming the first three columns are xyz

        # Estimate normals for FPFH
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Compute FPFH
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        
        return np.asarray(fpfh.data).T  # Return the FPFH features as numpy array
    
    def select_best_fpfh_features(self, all_fpfh_features):
        """
        Select the best FPFH features based on their discriminative power (variance) across the entire point cloud.
        Here, we select the features with the highest variance.
        """
        best_features = []
        
        # Calculate the variance of each feature across the entire point cloud
        for i in range(all_fpfh_features.shape[1]):
            feature = all_fpfh_features[:, i]  # Extract the i-th feature for all points
            feature_variance = np.var(feature)  # Compute variance of this feature
            best_features.append((feature_variance, i))  # Store variance and corresponding feature index

        # Sort features by variance (highest variance first) and select the top 5
        best_features.sort(reverse=True, key=lambda x: x[0])
        selected_feature_indices = [x[1] for x in best_features[:5]]  # Select the top 5 features based on variance

        return selected_feature_indices

    def normalize_features(self, features):
        """
        Normalize the FPFH features (e.g., Z-score or Min-Max normalization).
        """
        # Z-score normalization
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        features_normalized = (features - mean) / std
        
        # Alternatively, you can use Min-Max normalization:
        # min_val = np.min(features, axis=0)
        # max_val = np.max(features, axis=0)
        # features_normalized = (features - min_val) / (max_val - min_val)
        return features_normalized
    
    def add_fpfh(self, points):
        # Compute all FPFH features
        all_fpfh_features = self.compute_fpfh_features(points)

        # Select the best FPFH features based on their variance
        selected_fpfh_indices = self.select_best_fpfh_features(all_fpfh_features)
        
        # Keep only the selected FPFH features
        selected_fpfh_features = all_fpfh_features[:, selected_fpfh_indices]
        
        # Normalize the selected FPFH features
        selected_fpfh_features = self.normalize_features(selected_fpfh_features)
        
        # Now append the normalized selected FPFH features to the point cloud data
        points = np.hstack((points, selected_fpfh_features))  # Add the selected and normalized FPFH features to the point cloud data

        return points
        
    def norm(self, points, color=False, nir=False, intensity=False):
        if color == False and nir == False and intensity == False:
            min_pt, max_pt = np.min(points, axis=0), np.max(points, axis=0)
            maxXYZ = np.max(max_pt)
            minXYZ = np.min(min_pt)
            min_pt[:] = minXYZ
            max_pt[:] = maxXYZ
            centroid = np.mean(points, axis=0)
            points -= centroid
            max_distance = np.max(np.linalg.norm(points, axis=1))
            points /= max_distance
            points = points.astype(np.float32)
            max_pt = max_pt.astype(np.float32)
            pt = np.concatenate(( np.expand_dims(min_pt, 0),  np.expand_dims(max_pt, 0)), axis = 0)
        else:
            # 只对 points 的前三列 xyz 进行标准化
            xyz = points[:, :3]  # 提取前三列
            min_pt_xyz = np.min(xyz, axis=0)
            max_pt_xyz = np.max(xyz, axis=0)
            maxXYZ = np.max(max_pt_xyz)
            minXYZ = np.min(min_pt_xyz)
            min_pt_xyz[:] = minXYZ
            max_pt_xyz[:] = maxXYZ
            centroid_xyz = np.mean(xyz, axis=0)
            xyz -= centroid_xyz
            max_distance_xyz = np.max(np.linalg.norm(xyz, axis=1))
            xyz /= max_distance_xyz

            # 将标准化后的 xyz 替换回 points
            points[:, :3] = xyz
            
            # 对rgb标准化
            if color == True:
                color_data = points[:, 3:6]  # 提取 RGB 数据（假设 points 的 3:6 列是 RGB 值）
                color_data = color_data / 255.0  # 将 RGB 值从 [0, 255] 归一化到 [0, 1]
                color_data = (color_data - color_data.mean(axis=0)) / color_data.std(axis=0)  # 标准化（均值为 0，标准差为 1）
                points[:, 3:6] = color_data  # 将标准化后的 RGB 值写回 points
                
            # 转换为 float32 类型
            points = points.astype(np.float32)
            min_pt_xyz = min_pt_xyz.astype(np.float32)
            max_pt_xyz = max_pt_xyz.astype(np.float32)

            # 生成 pt
            pt = np.concatenate((np.expand_dims(min_pt_xyz, 0), np.expand_dims(max_pt_xyz, 0)), axis=0)

            # 更新 centroid 和 max_distance
            centroid = centroid_xyz
            max_distance = max_distance_xyz
        return points, pt, centroid, max_distance

    def compute_geometric_features(self, pcd, radius=0.1, k_neighbors=30):
        """
        计算点云的几何特征，包括曲率、线性度和平面度。
        输入：
            pcd: open3d.geometry.PointCloud 对象
            radius: 邻域搜索半径
            k_neighbors: 用于计算邻域的最近邻点数量
        输出：
            features: 每个点的特征数组 [curvature, linearity, planarity]
            scores: 每个点的重要性评分
        """
        points = np.asarray(pcd.points)
        n_points = points.shape[0]
        
        # 使用KD树加速邻域搜索
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # 初始化特征数组
        curvatures = np.zeros(n_points)
        linearities = np.zeros(n_points)
        planarities = np.zeros(n_points)
        
        for i in range(n_points):
            # 获取邻域点
            neighbors = points[indices[i]]
            
            # 计算协方差矩阵
            cov_matrix = np.cov(neighbors.T)
            
            # 特征值分解
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # 从大到小排序
            
            # 计算曲率、线性度、平面度
            lambda1, lambda2, lambda3 = eigenvalues
            total_variance = lambda1 + lambda2 + lambda3 + 1e-10  # 避免除以零
            
            curvatures[i] = lambda3 / total_variance
            linearities[i] = (lambda1 - lambda2) / (lambda1 + 1e-10)
            planarities[i] = (lambda2 - lambda3) / (lambda1 + 1e-10)
        
        # 计算重要性评分
        w_c, w_l, w_p = 0.4, 0.4, 0.2  # 权重，可调优
        scores = w_c * curvatures + w_l * linearities - w_p * planarities
        
        # 归一化评分到 [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        features = np.vstack((curvatures, linearities, planarities)).T
        return features, scores

    def feature_aware_probability_sampling(self, pcd, scores, n_init=4096, beta=5.0):
        """
        基于特征重要性评分的概率采样。
        输入：
            pcd: open3d.geometry.PointCloud 对象
            scores: 每个点的重要性评分
            n_init: 初始采样点数
            beta: 控制概率分布的参数
        输出：
            initial_indices: 初始采样点的索引
        """
        points = np.asarray(pcd.points)
        n_points = points.shape[0]
        
        # 动态调整 n_init，确保不超过 n_points
        n_init = min(n_init, n_points)
        
        # 计算采样概率
        probs = np.exp(beta * scores) / np.sum(np.exp(beta * scores))
        
        # 加权随机采样
        initial_indices = np.random.choice(n_points, size=n_init, replace=False, p=probs)
        
        return initial_indices

    def farthest_point_sampling(self, points, n_sample, existing_indices=None):
        """
        基于最远点采样的均匀采样。
        输入：
            points: 点云数据 (N, 3)
            n_sample: 目标采样点数
            existing_indices: 已选点的索引（可选）
        输出：
            sampled_indices: 采样点的索引
        """
        n_points = points.shape[0]
        
        # 动态调整 n_sample，确保不超过 n_points
        n_sample = min(n_sample, n_points)
        
        sampled_indices = []
        
        if existing_indices is not None:
            sampled_indices = list(existing_indices)
        else:
            # 随机选择第一个点
            sampled_indices.append(np.random.randint(n_points))
        
        while len(sampled_indices) < n_sample:
            sampled_points = points[sampled_indices]
            remaining_indices = np.setdiff1d(np.arange(n_points), sampled_indices)
            remaining_points = points[remaining_indices]
            
            # 计算剩余点到已选点的最小距离
            distances = cdist(remaining_points, sampled_points)
            min_distances = np.min(distances, axis=1)
            
            # 选择距离最大的点
            next_index = remaining_indices[np.argmax(min_distances)]
            sampled_indices.append(next_index)
        
        return np.array(sampled_indices)

    def feature_aware_adaptive_sampling(self, pcd, target_points=2048, radius=0.1, k_neighbors=30, beta=5.0):
        """
        完整的特征感知自适应采样方法。
        输入：
            pcd: open3d.geometry.PointCloud 对象
            target_points: 目标采样点数 (默认 2048)
            radius: 邻域搜索半径
            k_neighbors: 邻域点数
            beta: 概率分布控制参数
        输出：
            sampled_pcd: 采样后的点云
            sampled_indices: 采样点的索引
        """
        points = np.asarray(pcd.points)
        n_points = points.shape[0]
        
        # 如果目标点数超过总点数，直接返回原始点云
        # if target_points >= n_points:
        #     print(f"Warning: target_points ({target_points}) >= n_points ({n_points}). Returning original point cloud.")
        #     return pcd, np.arange(n_points)
        
        # Step 1: 计算几何特征和重要性评分
        features, scores = self.compute_geometric_features(pcd, radius, k_neighbors)
        
        # Step 2: 初始概率采样
        initial_indices = self.feature_aware_probability_sampling(pcd, scores, n_init=4096, beta=beta)
        initial_pcd = pcd.select_by_index(initial_indices)
        
        # Step 3: 特征优先精炼
        initial_points = np.asarray(initial_pcd.points)
        initial_scores = scores[initial_indices]
        
        # 按重要性评分排序，保留前 80% 的点
        n_priority = int(target_points * 0.8)
        priority_indices = np.argsort(initial_scores)[::-1][:n_priority]
        priority_indices_global = initial_indices[priority_indices]
        
        # Step 4: 空间均匀性补充
        n_uniform = target_points - n_priority
        final_indices = self.farthest_point_sampling(np.asarray(pcd.points), target_points, priority_indices_global)
        
        # Step 5: 生成采样点云
        sampled_pcd = pcd.select_by_index(final_indices)
        
        return sampled_pcd, final_indices



    def __getitem__(self, item):
        file_path = self.file_list[item]
        frame_id = file_path.split('/')[-1]
        file_form = file_path.split('.')[-1]
        if file_form == 'ply':
            points = read_ply(file_path)
        elif file_form == 'xyz':
            points = read_pts(file_path, self.color, self.nir, self.intensity)
        else:
            print("none support file form")
        if self.transform is not None:
            points = self.transform(points)


        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        # sampled_pcd, sampled_indices = self.feature_aware_adaptive_sampling(
        #     pcd, target_points=2048, radius=0.1, k_neighbors=30, beta=5.0
        # )
        if len(points) > self.npoint:
            idx = np.random.randint(0, len(points), self.npoint)
        else:
            idx = np.random.randint(0, len(points), self.npoint - len(points))
            idx = np.append(np.arange(0, len(points)), idx)
        np.random.shuffle(idx)


        points = points[idx]
        # points = points[sampled_indices]
        points, pt, centroid, max_distance = self.norm(points, self.color, self.nir, self.intensity)
        if self.fpfh:
            points = self.add_fpfh(points)
        data_dict = {'points': points, 'frame_id': frame_id, 'minMaxPt': pt, 'centroid': centroid, 'max_distance': max_distance}
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key == 'points':
                    ret[key] = np.concatenate(val, axis=0).reshape([batch_size, -1, val[0].shape[-1]])
                elif key in ['frame_id']:
                    ret[key] = val
                elif key in ['minMaxPt']:
                    ret[key] = val
                elif key in ['centroid']:
                    ret[key] = val
                elif key in ['max_distance']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret




