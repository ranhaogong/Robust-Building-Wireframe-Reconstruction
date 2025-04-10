import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import os
import shutil
import open3d as o3d
import torch
from torch_cluster import knn
import torch.sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def compute_mrgd_features(points, device='cuda'):
    """保留谱分析的优化 mrgd 计算"""
    # 输入转为 GPU 张量
    coord = torch.from_numpy(points[:, :3]).float().to(device)
    N = coord.size(0)
    batch = torch.zeros(N, dtype=torch.long, device=device)

    # 单次 k-NN 计算 (k=32)
    edge_l = knn(coord, coord, k=32, batch_x=batch, batch_y=batch)  # [2, N*32]
    edge_l = edge_l[:, :N * 32]  # 强制截取
    edge_s = edge_l[:, ::4][:,:N * 8]  # 从 k=32 提取 k=8

    # 计算法向量
    n_s = compute_normals_torch(coord, edge_s[1], k=8)
    n_l = compute_normals_torch(coord, edge_l[1], k=32)

    # 曲率
    k_s = compute_curvature(coord, edge_s[1], n_s, k=8)
    k_l = compute_curvature(coord, edge_l[1], n_l, k=32)

    # 高度方差
    z = coord[:, 2]
    sigma_s = torch.sqrt(((z[edge_s[1]] - z[edge_s[0]]) ** 2).view(N, 8).mean(dim=1))
    sigma_l = torch.sqrt(((z[edge_l[1]] - z[edge_l[0]]) ** 2).view(N, 32).mean(dim=1))

    # 谱特征（优化后的逐点谱分析）
    S_s = compute_graph_spectral_torch(coord, edge_s, k=8, m=3)
    S_l = compute_graph_spectral_torch(coord, edge_l, k=32, m=3)

    # 非线性对比核
    sigma = 0.1
    K_c = torch.exp(-((k_s - k_l) ** 2) / (2 * sigma ** 2))
    K_h = torch.exp(-((sigma_s - sigma_l) ** 2) / (2 * sigma ** 2))
    K_s = torch.exp(-((S_s - S_l) ** 2) / (2 * sigma ** 2))

    # mrgd 特征
    mrgd = torch.stack([K_c * k_s, K_h * sigma_s, K_s * S_s], dim=1)  # [N, 3]

    # Min-Max 归一化
    mrgd_min = mrgd.min(dim=0, keepdim=True)[0]
    mrgd_max = mrgd.max(dim=0, keepdim=True)[0]
    mrgd_normalized = (mrgd - mrgd_min) / (mrgd_max - mrgd_min + 1e-8)

    return mrgd_normalized.cpu().numpy()

def compute_normals_torch(coord, indices, k):
    """PyTorch 原生法向量计算"""
    N = coord.size(0)
    points = coord[indices].view(N, k, 3)  # [N, k, 3]
    centered = points - points.mean(dim=1, keepdim=True)  # [N, k, 3]
    cov = torch.bmm(centered.transpose(1, 2), centered) / k  # [N, 3, 3]
    _, eig_vecs = torch.linalg.eigh(cov)  # [N, 3, 3]
    normals = eig_vecs[:, :, 0]  # 最小特征向量 [N, 3]
    normals = normals * torch.sign((normals * coord).sum(dim=1, keepdim=True))
    return normals

def compute_curvature(coord, indices, normals, k):
    """简化的曲率计算"""
    N = coord.size(0)
    points = coord[indices].view(N, k, 3)  # [N, k, 3]
    vectors = points - coord.unsqueeze(1)  # [N, k, 3]
    proj_dist = (vectors * normals.unsqueeze(1)).sum(dim=-1)  # [N, k]
    curvature = proj_dist.abs().mean(dim=1)  # [N]
    return curvature / (curvature.max() + 1e-8)

def compute_graph_spectral_torch(coord, edge_index, k, m=3):
    """GPU 加速的逐点谱分析"""
    N = coord.size(0)
    row, col = edge_index

    # 构建稀疏邻接矩阵（GPU）
    values = torch.ones(row.size(0), device=coord.device, dtype=torch.float)
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), values, (N, N))
    
    # 计算度矩阵
    degree = torch.sparse.sum(adj, dim=1).to_dense()  # [N]
    D = torch.sparse_coo_tensor(
        torch.stack([torch.arange(N, device=coord.device), torch.arange(N, device=coord.device)]),
        degree,
        (N, N)
    )

    # 拉普拉斯矩阵 L = D - A
    L = D - adj

    # 近似特征值分解（使用幂迭代法取前 m 个特征值）
    def power_iteration(L, num_iter=10):
        v = torch.randn(N, m, device=coord.device)
        v = v / torch.norm(v, dim=0, keepdim=True)
        for _ in range(num_iter):
            v = L @ v  # 稀疏矩阵乘法
            v = v / (torch.norm(v, dim=0, keepdim=True) + 1e-8)
        return v

    eigenvectors = power_iteration(L)  # [N, m]
    
    # 计算谱能量（逐点）
    spectral_energy = torch.sum(eigenvectors ** 2, dim=1)  # [N]
    return spectral_energy / (spectral_energy.max() + 1e-8)

def compute_fpfh_features(points):
    # Convert points to open3d format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # assuming the first three columns are xyz

    # Estimate normals for FPFH
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Compute FPFH
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    
    return np.asarray(fpfh.data).T  # Return the FPFH features as numpy array

def select_best_fpfh_features(all_fpfh_features):
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

def normalize_features(features):
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

# 文件路径
input_file = "/data/haoran/dataset/building3d/roof/Tallinn/train/xyz_norm/46890.xyz"
output_dir = "/data/haoran/Point2Roof/vis_fpfh_mrgd"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取点云文件（仅 xyz）
# points = np.loadtxt(input_file)[:, :3]  # [n, 3]
points = read_pts(input_file)
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
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 检查点云是否正确加载
if points.shape[0] == 0:
    raise ValueError(f"Failed to load point cloud from {input_file}")

# 估计法向量（FPFH 需要）
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
normals = np.asarray(pcd.normals)

# 计算 FPFH 和 MRGD（调用你的实现）
# 假设你的函数签名如下：
# Compute all FPFH features
all_fpfh_features = compute_fpfh_features(points)

# Select the best FPFH features based on their variance
selected_fpfh_indices =select_best_fpfh_features(all_fpfh_features)

# Keep only the selected FPFH features
selected_fpfh_features = all_fpfh_features[:, selected_fpfh_indices]

# Normalize the selected FPFH features
fpfh = normalize_features(selected_fpfh_features)
mrgd = compute_mrgd_features(points)           # [n, 3]，你的实现

# 计算标量值用于颜色映射
# fpfh_norm = np.linalg.norm(fpfh, axis=1)  # FPFH 范数
# mrgd_norm = np.linalg.norm(mrgd, axis=1)  # MRGD 范数
fpfh_norm = np.linalg.norm(fpfh[:, 0])  # FPFH 范数
mrgd_norm = np.linalg.norm(mrgd[:, 2])  # MRGD 范数

# 创建颜色映射
def create_colors(norm_values, is_fpfh=True):
    colors = np.zeros((len(norm_values), 3))
    norm_values = norm_values / norm_values.max()  # 归一化到 [0, 1]
    if is_fpfh:
        colors[:, 0] = norm_values  # 红色通道 (FPFH)
        colors[:, 1] = 1 - norm_values  # 黄色渐变
    else:
        colors[:, 2] = norm_values  # 蓝色通道 (MRGD)
        colors[:, 1] = 1 - norm_values  # 绿色渐变
    return colors

# FPFH 点云
fpfh_colors = create_colors(fpfh_norm, is_fpfh=True)
pcd_fpfh = o3d.geometry.PointCloud()
pcd_fpfh.points = o3d.utility.Vector3dVector(points)
pcd_fpfh.colors = o3d.utility.Vector3dVector(fpfh_colors)

# MRGD 点云
mrgd_colors = create_colors(mrgd_norm, is_fpfh=False)
pcd_mrgd = o3d.geometry.PointCloud()
pcd_mrgd.points = o3d.utility.Vector3dVector(points)
pcd_mrgd.colors = o3d.utility.Vector3dVector(mrgd_colors)

# # 可视化并保存
# fig = plt.figure(figsize=(12, 5))

# # FPFH 可视化
# ax1 = fig.add_subplot(121, projection='3d')
# scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=fpfh_norm, cmap='Reds', s=2)
# ax1.view_init(elev=90, azim=0)  # 俯视视角
# ax1.set_title("FPFH Feature")
# plt.colorbar(scatter1, ax=ax1)

# # MRGD 可视化
# ax2 = fig.add_subplot(122, projection='3d')
# scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=mrgd_norm, cmap='Blues', s=2)
# ax2.view_init(elev=90, azim=0)  # 俯视视角
# ax2.set_title("MRGD Feature")
# plt.colorbar(scatter2, ax=ax2)

# # 保存
# plt.savefig(os.path.join(output_dir, "fpfh_mrgd_visual.png"), dpi=300, bbox_inches='tight')
# plt.close()

# print(f"Visualization saved to {output_dir}/fpfh_mrgd_visual.png")

# 保存带颜色的点云为 .xyz 文件
def save_colored_point_cloud(pcd, output_path):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255  # 转换为 [0, 255]
    data = np.hstack((points, colors))
    np.savetxt(output_path, data, fmt='%.6f %.6f %.6f %.0f %.0f %.0f', header='x y z r g b')

# 保存 FPFH 和 MRGD 点云
save_colored_point_cloud(pcd_fpfh, os.path.join(output_dir, "fpfh_colored.xyz"))
save_colored_point_cloud(pcd_mrgd, os.path.join(output_dir, "mrgd_colored.xyz"))

print(f"Colored point clouds saved to {output_dir}/fpfh_colored.xyz and {output_dir}/mrgd_colored.xyz")