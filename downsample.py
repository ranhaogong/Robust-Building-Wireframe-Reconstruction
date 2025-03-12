import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import os
from pathlib import Path
from tqdm import tqdm  # 导入 tqdm 库

# 设置随机种子以确保可重复性
np.random.seed(42)

def normalize_point_cloud(pcd):
    """
    对点云进行归一化：平移到原点并缩放到单位范围 [-1, 1]^3。
    输入：
        pcd: open3d.geometry.PointCloud 对象
    输出：
        normalized_pcd: 归一化后的点云
        center: 平移中心
        scale: 缩放因子
    """
    points = np.asarray(pcd.points)
    
    # 平移到原点
    center = np.mean(points, axis=0)
    points_centered = points - center
    
    # 缩放到单位范围
    max_range = np.max(np.abs(points_centered))
    scale = max_range if max_range > 0 else 1.0  # 避免除以零
    points_normalized = points_centered / scale
    
    # 更新点云
    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(points_normalized)
    
    return normalized_pcd, center, scale

def denormalize_point_cloud(normalized_pcd, center, scale):
    """
    将归一化后的点云反归一化到原始尺度。
    输入：
        normalized_pcd: 归一化后的点云
        center: 平移中心
        scale: 缩放因子
    输出：
        denormalized_pcd: 反归一化后的点云
    """
    points_normalized = np.asarray(normalized_pcd.points)
    points_denormalized = points_normalized * scale + center
    
    denormalized_pcd = o3d.geometry.PointCloud()
    denormalized_pcd.points = o3d.utility.Vector3dVector(points_denormalized)
    
    return denormalized_pcd

def read_xyz_point_cloud(file_path):
    """
    读取 .xyz 点云文件，仅提取前三列 (x, y, z) 作为坐标。
    输入：
        file_path: .xyz 文件路径
    输出：
        pcd: open3d.geometry.PointCloud 对象
    """
    # 读取 .xyz 文件
    data = np.loadtxt(file_path, delimiter=' ')  # 假设空格分隔
    points = data[:, :3]  # 仅提取前三列 (x, y, z)
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

def compute_geometric_features(pcd, radius=0.1, k_neighbors=30):
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

def feature_aware_probability_sampling(pcd, scores, n_init=4096, beta=5.0):
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

def farthest_point_sampling(points, n_sample, existing_indices=None):
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

def feature_aware_adaptive_sampling(pcd, target_points=2048, radius=0.1, k_neighbors=30, beta=5.0):
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
    if target_points >= n_points:
        print(f"Warning: target_points ({target_points}) >= n_points ({n_points}). Returning original point cloud.")
        return pcd, np.arange(n_points)
    
    # Step 1: 计算几何特征和重要性评分
    features, scores = compute_geometric_features(pcd, radius, k_neighbors)
    
    # Step 2: 初始概率采样
    initial_indices = feature_aware_probability_sampling(pcd, scores, n_init=4096, beta=beta)
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
    final_indices = farthest_point_sampling(np.asarray(pcd.points), target_points, priority_indices_global)
    
    # Step 5: 生成采样点云
    sampled_pcd = pcd.select_by_index(final_indices)
    
    return sampled_pcd, final_indices

def write_xyz_point_cloud(file_path, sampled_pcd, original_data, sampled_indices):
    """
    将采样后的点云保存为 .xyz 格式，保留原始点的所有信息（包括 RGB 等）。
    输入：
        file_path: 保存路径
        sampled_pcd: 采样后的点云对象
        original_data: 原始点云的完整数据 (包括 RGB 等)
        sampled_indices: 采样点的索引
    """
    sampled_data = original_data[sampled_indices]
    np.savetxt(file_path, sampled_data, delimiter=' ', fmt='%.6f')


def random_sampling(pcd, num_samples=2048):
    """
    随机从点云中采样指定数量的点。
    输入：
        pcd: open3d.geometry.PointCloud 对象
        num_samples: 目标采样点数
    输出：
        sampled_pcd: 采样后的点云对象
    """
    points = np.asarray(pcd.points)
    n_points = points.shape[0]
    
    sampled_indices = np.random.choice(n_points, num_samples, replace=False)

    sampled_pcd = pcd.select_by_index(sampled_indices)
    return sampled_pcd, sampled_indices

def uniform_sampling(pcd, target_points=2048):
    """
    使用 Open3D 的 API 实现均匀采样到目标点数。
    输入：
        pcd: open3d.geometry.PointCloud 对象
        target_points: 目标采样点数 (默认 2048)
    输出：
        sampled_pcd: 采样后的点云
        sampled_indices: 采样点的索引
    """
    points = np.asarray(pcd.points)
    n_points = points.shape[0]
    
    # 使用 Open3D 的均匀采样 API
    sampled_pcd = pcd.uniform_down_sample(every_k_points=n_points // target_points)
    
    # 如果采样点数不精确匹配 target_points，调整到目标点数
    sampled_points = np.asarray(sampled_pcd.points)
    current_points = sampled_points.shape[0]
    
    if current_points < target_points:
        # 如果采样点数不足，随机补充
        additional_indices = np.random.choice(n_points, size=target_points - current_points, replace=False)
        additional_points = points[additional_indices]
        final_points = np.vstack((sampled_points, additional_points))
        sampled_pcd.points = o3d.utility.Vector3dVector(final_points)
        sampled_indices = np.concatenate((np.arange(current_points), additional_indices))
    elif current_points > target_points:
        # 如果采样点数过多，随机裁剪
        sampled_indices = np.random.choice(current_points, size=target_points, replace=False)
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points[sampled_indices])
    else:
        sampled_indices = np.arange(target_points)
    
    return sampled_pcd, sampled_indices

def fps_sampling(points, num_samples=2048):
    """
    最远点采样（FPS）
    """
    points = np.asarray(pcd.points)
    num_points = points.shape[0]
    if num_points <= num_samples:
        return np.arange(num_points)

    sampled_indices = [np.random.randint(num_points)]
    
    for _ in range(num_samples - 1):
        sampled_points = points[sampled_indices]
        remaining_indices = np.setdiff1d(np.arange(num_points), sampled_indices)
        remaining_points = points[remaining_indices]
        
        distances = cdist(remaining_points, sampled_points)
        min_distances = np.min(distances, axis=1)
        next_index = remaining_indices[np.argmax(min_distances)]
        
        sampled_indices.append(next_index)
        
    sampled_pcd = pcd.select_by_index(sampled_indices)
    return sampled_pcd, sampled_indices

def downsample(filename, outputdir, pcd, target_points=2048):
    # 归一化点云
    # normalized_pcd, center, scale = normalize_point_cloud(pcd)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # print(f"归一化后的点云范围: {np.min(pcd.points, axis=0)} 到 {np.max(pcd.points, axis=0)}")
    # 执行采样（基于归一化后的点云）

    sampled_pcd, sampled_indices = feature_aware_adaptive_sampling(
        pcd, target_points=2048, radius=0.1, k_neighbors=30, beta=5.0
    )
    # 随机采样
    random_pcd, random_pcd_idx = random_sampling(pcd, target_points)
    # 均匀采样
    uniform_pcd, uniform_pcd_idx = uniform_sampling(pcd, target_points)
    # fps采样
    fps_pcd, fps_pcd_idx = fps_sampling(pcd, target_points)
    # print(f"采样后点云点数: {len(sampled_normalized_pcd.points)}")
    
    # 反归一化采样点云
    # sampled_pcd = denormalize_point_cloud(sampled_normalized_pcd, center, scale)
    # random_sampled_pcd = denormalize_point_cloud(random_pcd, center, scale)
    # uniform_sampled_pcd = denormalize_point_cloud(uniform_pcd, center, scale)
    # fps_sampled_pcd = denormalize_point_cloud(fps_pcd, center, scale)
    
    
    # 保存采样点云（保留原始格式，包括 RGB 等信息）
    output_filename = filename + "_sampled_roof_point_cloud.xyz"
    random_output_filename = filename + "_random_sampled_roof_point_cloud.xyz"
    uniform_output_filename = filename + "_uniform_sampled_roof_point_cloud.xyz"
    fps_output_filename = filename + "_fps_sampled_roof_point_cloud.xyz"
    output_path = Path(outputdir) / output_filename
    random_output_path = Path(outputdir) / random_output_filename
    uniform_output_path = Path(outputdir) / uniform_output_filename
    fps_output_path = Path(outputdir) / fps_output_filename
    write_xyz_point_cloud(output_path, sampled_pcd, original_data, sampled_indices)
    write_xyz_point_cloud(random_output_path, random_pcd, original_data, random_pcd_idx)
    write_xyz_point_cloud(uniform_output_path, uniform_pcd, original_data, uniform_pcd_idx)
    write_xyz_point_cloud(fps_output_path, fps_pcd, original_data, fps_pcd_idx)
    print(f"采样点云已保存至: {output_path}")
    print(f"随机采样点云已保存至: {random_output_path}")
    print(f"均匀采样点云已保存至: {uniform_output_path}")
    print(f"fps采样点云已保存至: {fps_output_path}")
    

# 示例使用
if __name__ == "__main__":
    # 读取 .xyz 点云文件（替换为你的点云文件路径）
    # pcd_path = "/data/haoran/dataset/building3d/roof/Tallinn/train/xyz_norm/2.xyz"  # 请替换为实际路径
    pcd_files_path = "/data/haoran/dataset/building3d/roof/Tallinn/train/xyz_norm"
    pcd_files = [f for f in os.listdir(pcd_files_path) if f.endswith('.xyz')]
    pth = r"/data/haoran/Point2Roof/vis_sample" # dir1不存在
    if not os.path.exists(pth):
        os.mkdir(pth)
    for f in tqdm(pcd_files, desc="处理文件中"):
        f = os.path.join(pcd_files_path, f)
        filename = f.split('/')[-1].split('.')[0]
        original_data = np.loadtxt(f, delimiter=' ')  # 读取完整数据以便保存
        pcd = read_xyz_point_cloud(f)
        if np.asarray(pcd.points).shape[0] < 2048:
            continue
        # print(f"原始点云点数: {len(pcd.points)}")
        downsample(filename, pth, pcd, 2048)
    
    
    

    