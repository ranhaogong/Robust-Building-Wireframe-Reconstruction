import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import os
import shutil
import open3d as o3d

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


class Building3DDataset(Dataset):
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

    def transform_all(self, points, color=False, nir=False, intensity=False, fpfh=False):
        if color == False and nir == False and intensity == False and fpfh == False:
            # 假设 points 是一个 N x 3 的数组，只有 x, y, z 坐标
            xyz = points[:, :3]  # 提取前三列（实际上就是全部数据）
            xyz_transformed = self.transform(xyz)  # 对 x, y, z 进行变换

            # 直接将变换后的 x, y, z 坐标赋值给 points
            points = xyz_transformed
            
        else:
            # 假设 points 是一个 N x M 的数组，其中前三列是 x, y, z 坐标
            xyz = points[:, :3]  # 提取前三列
            xyz_transformed = self.transform(xyz)  # 对 x, y, z 进行变换

            # 将变换后的 x, y, z 坐标与原始数据的其他列合并
            points_transformed = np.hstack((xyz_transformed, points[:, 3:]))

            # 更新 points
            points = points_transformed
        
        return points
    
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
    
    def norm(self, points, vectors, color=False, nir=False, intensity=False):
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
            vectors -= centroid
            vectors /= max_distance
            points = points.astype(np.float32)
            vectors = vectors.astype(np.float32)
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

            # 对 vectors 进行同样的标准化
            vectors -= centroid_xyz
            vectors /= max_distance_xyz
            
            # 对rgb标准化
            if color == True:
                color_data = points[:, 3:6]  # 提取 RGB 数据（假设 points 的 3:6 列是 RGB 值）
                color_data = color_data / 255.0  # 将 RGB 值从 [0, 255] 归一化到 [0, 1]
                color_data = (color_data - color_data.mean(axis=0)) / color_data.std(axis=0)  # 标准化（均值为 0，标准差为 1）
                points[:, 3:6] = color_data  # 将标准化后的 RGB 值写回 points

            # 转换为 float32 类型
            points = points.astype(np.float32)
            vectors = vectors.astype(np.float32)
            min_pt_xyz = min_pt_xyz.astype(np.float32)
            max_pt_xyz = max_pt_xyz.astype(np.float32)

            # 生成 pt
            pt = np.concatenate((np.expand_dims(min_pt_xyz, 0), np.expand_dims(max_pt_xyz, 0)), axis=0)

            # 更新 centroid 和 max_distance
            centroid = centroid_xyz
            max_distance = max_distance_xyz

        return points, vectors, pt, centroid, max_distance
    
    def __getitem__(self, item):
        file_path = self.file_list[item]
        frame_id = file_path.split('/')[-1]
        points = read_pts(file_path + '/points.xyz', self.color, self.nir, self.intensity)
        points = self.transform_all(points, self.color, self.nir, self.intensity, self.fpfh)
        if len(points) > self.npoint:
            idx = np.random.randint(0, len(points), self.npoint)
        else:
            idx = np.random.randint(0, len(points), self.npoint - len(points))
            idx = np.append(np.arange(0, len(points)), idx)
        np.random.shuffle(idx)
        points = points[idx]
        vectors, edges = load_obj(self.file_list[item] + '/polygon.obj')
        
        points, vectors, pt, centroid, max_distance = self.norm(points, vectors, self.color, self.nir, self.intensity)
        if self.fpfh:
            points = self.add_fpfh(points)
        data_dict = {'points': points, 'vectors': vectors, 'edges': edges, 'frame_id': frame_id, 'minMaxPt': pt, 'centroid': centroid, 'max_distance': max_distance}
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
                elif key in ['vectors', 'edges']:
                    max_vec = max([len(x) for x in val])
                    batch_vecs = np.ones((batch_size, max_vec, val[0].shape[-1]), dtype=np.float32) * -1e1
                    for k in range(batch_size):
                        batch_vecs[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_vecs
                elif key in ['frame_id']:
                    ret[key] = val
                elif key in ['minMaxPt']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret




