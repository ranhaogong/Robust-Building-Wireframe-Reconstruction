import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import random
import shutil

def set_random_seed(seed=42):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    o3d.utility.random.seed(seed)

def load_xyz(file_path):
    """加载 .xyz 文件，包含颜色信息"""
    data = np.loadtxt(file_path)
    points = data[:, :3]  # 前三列为 (x, y, z)
    colors = data[:, 3:] if data.shape[1] > 3 else None  # 后三列为 (r, g, b)，如果存在
    return points, colors

def save_xyz(file_path, points, colors=None):
    """保存点云到 .xyz 文件，包含颜色信息（如果提供）"""
    if colors is not None:
        data = np.hstack((points, colors))
    else:
        data = points
    np.savetxt(file_path, data, fmt='%.6f')

def normalize_point_cloud(points):
    """
    对点云进行归一化：平移到原点并缩放到单位范围 [-1, 1]^3。
    """
    center = np.mean(points, axis=0)
    points_centered = points - center
    max_range = np.max(np.abs(points_centered))
    scale = max_range if max_range > 0 else 1.0  # 避免除以零
    points_normalized = points_centered / scale
    return points_normalized, center, scale

def denormalize_point_cloud(points, center, scale):
    """
    将归一化后的点云反归一化到原始尺度。
    """
    points_denormalized = points * scale + center
    return points_denormalized

def enhanced_euclidean_clustering(pcd, distance_threshold=0.03, height_threshold=0.05):
    """
    增强的欧几里得聚类，结合空间距离和高度差异，确保所有点都被分配
    """
    points = np.asarray(pcd.points)
    n_points = len(points)
    
    # 构建 KD 树
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    visited = np.zeros(n_points, dtype=bool)
    clusters = []
    
    for i in range(n_points):
        if visited[i]:
            continue
        
        current_cluster = [i]
        visited[i] = True
        seed_queue = [i]
        
        while seed_queue:
            seed_idx = seed_queue.pop(0)
            seed_point = points[seed_idx]
            
            # 查找邻域点
            [k, idx, dist] = kdtree.search_radius_vector_3d(seed_point, distance_threshold)
            for neighbor_idx in idx:
                if visited[neighbor_idx]:
                    continue
                
                neighbor_point = points[neighbor_idx]
                
                # 计算高度差异（z 坐标差异）
                height_diff = abs(seed_point[2] - neighbor_point[2])
                
                # 如果空间距离和高度差异都满足条件，则加入当前簇
                if height_diff < height_threshold:
                    visited[neighbor_idx] = True
                    current_cluster.append(neighbor_idx)
                    seed_queue.append(neighbor_idx)
        
        cluster_points = points[current_cluster]
        clusters.append((cluster_points, current_cluster))  # 返回点云和原始索引
    
    return clusters

def merge_small_clusters(clusters, min_points=20):
    """
    将点数小于 min_points 的簇中的每个点单独归并到最近的簇中
    """
    if not clusters:
        return clusters
    
    # 分离大簇和小簇
    large_clusters = [(c, idx) for c, idx in clusters if len(c) >= min_points]
    small_clusters = [(c, idx) for c, idx in clusters if len(c) < min_points]
    
    if not small_clusters or not large_clusters:
        return clusters  # 如果没有小簇或没有大簇，直接返回
    
    # 为每个大簇构建 KD 树
    large_cluster_kdtrees = []
    for c, _ in large_clusters:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(c)
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        large_cluster_kdtrees.append(kdtree)
    
    # 存储合并后的簇（使用列表而不是元组）
    merged_clusters = [[c.copy(), idx.copy()] for c, idx in large_clusters]
    
    # 对每个小簇中的每个点，找到最近的大簇
    for small_cluster, small_indices in small_clusters:
        for point, orig_idx in zip(small_cluster, small_indices):
            min_dist = float('inf')
            nearest_cluster_idx = -1
            
            # 计算点到每个大簇中最近点的距离
            for i, kdtree in enumerate(large_cluster_kdtrees):
                [k, idx, dist] = kdtree.search_knn_vector_3d(point, 1)
                if dist[0] < min_dist:
                    min_dist = dist[0]
                    nearest_cluster_idx = i
            
            # 将点和原始索引添加到最近的大簇中
            merged_clusters[nearest_cluster_idx][0] = np.vstack([merged_clusters[nearest_cluster_idx][0], point])
            merged_clusters[nearest_cluster_idx][1].append(orig_idx)
    
    return merged_clusters

def generate_distinct_colors(n_clusters):
    """为每个簇生成不同的颜色（0-255 格式）"""
    hsv_colors = np.zeros((n_clusters, 3))
    hsv_colors[:, 0] = np.linspace(0, 360, n_clusters)  # 使用固定的颜色分布
    hsv_colors[:, 1] = 1.0
    hsv_colors[:, 2] = 1.0
    
    rgb_colors = np.zeros((n_clusters, 3), dtype=int)
    for i in range(n_clusters):
        h, s, v = hsv_colors[i]
        h /= 60.0
        c = v * s
        x = c * (1 - abs(h % 2 - 1))
        m = v - c
        
        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        rgb_colors[i] = [(r + m) * 255, (g + m) * 255, (b + m) * 255]
    
    return rgb_colors

def visualize_point_cloud(points_with_colors):
    """可视化带颜色的点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_with_colors[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points_with_colors[:, 3:] / 255.0)
    o3d.visualization.draw_geometries([pcd])

def process_xyz_files(input_dir, output_dir, distance_threshold=0.03, height_threshold=0.05, min_points=20, point_threshold=10000):
    """处理所有 .xyz 文件并进行单体化，总点数超过 point_threshold 才聚类"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    xyz_files = [f for f in os.listdir(input_dir) if f.endswith('.xyz')]
    
    for xyz_file in tqdm(xyz_files, desc="Processing files"):
        input_path = os.path.join(input_dir, xyz_file)
        points, input_colors = load_xyz(input_path)  # 加载原始点云和颜色
        
        # 检查总点数
        total_points = len(points)
        base_name = os.path.splitext(xyz_file)[0]
        
        if total_points <= point_threshold:
            # 点数不超过 threshold，直接保存原始文件
            output_file = os.path.join(output_dir, f"{base_name}_original.xyz")
            save_xyz(output_file, points, input_colors)
            continue
        
        # 点数超过 threshold，进行聚类
        points_normalized, center, scale = normalize_point_cloud(points)
        
        # 转换为 open3d 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_normalized)
        
        # 使用增强的欧几里得聚类
        initial_clusters = enhanced_euclidean_clustering(pcd, distance_threshold=distance_threshold, 
                                                       height_threshold=height_threshold)
        
        if len(initial_clusters) == 0:
            print(f"Warning: No clusters found for {xyz_file}")
            continue
        
        # 合并小簇
        final_clusters = merge_small_clusters(initial_clusters, min_points=min_points)
        
        if len(final_clusters) == 0:
            print(f"Warning: No clusters found after merging for {xyz_file}")
            continue
        
        # 为可视化生成不同的颜色
        vis_colors = generate_distinct_colors(len(final_clusters))
        
        # 合并所有簇到一个点云中，用于可视化（使用生成的颜色）
        merged_points_with_colors = []
        for i, (cluster, _) in enumerate(final_clusters):
            cluster_denormalized = denormalize_point_cloud(cluster, center, scale)
            cluster_colors = np.tile(vis_colors[i], (len(cluster_denormalized), 1))
            cluster_with_colors = np.hstack((cluster_denormalized, cluster_colors))
            merged_points_with_colors.append(cluster_with_colors)
        
        merged_points_with_colors = np.vstack(merged_points_with_colors)
        
        # 保存合并后的点云（使用生成的颜色）
        # output_file = os.path.join(output_dir, f"{base_name}_merge.xyz")
        # save_xyz(output_file, merged_points_with_colors[:, :3], merged_points_with_colors[:, 3:])
        
        # 单独保存每个簇到单独的 .xyz 文件（使用原始颜色）
        if input_colors is not None:
            for i, (cluster, orig_indices) in enumerate(final_clusters):
                cluster_denormalized = denormalize_point_cloud(cluster, center, scale)
                # 根据原始索引提取原始颜色
                cluster_colors = input_colors[orig_indices]
                cluster_with_colors = np.hstack((cluster_denormalized, cluster_colors))
                output_file = os.path.join(output_dir, f"{base_name}_cluster_{i}.xyz")
                save_xyz(output_file, cluster_with_colors[:, :3], cluster_with_colors[:, 3:])
        else:
            # 如果输入没有颜色，则使用生成的颜色
            for i, (cluster, _) in enumerate(final_clusters):
                cluster_denormalized = denormalize_point_cloud(cluster, center, scale)
                cluster_colors = np.tile(vis_colors[i], (len(cluster_denormalized), 1))
                cluster_with_colors = np.hstack((cluster_denormalized, cluster_colors))
                output_file = os.path.join(output_dir, f"{base_name}_cluster_{i}.xyz")
                save_xyz(output_file, cluster_with_colors[:, :3], cluster_with_colors[:, 3:])

def main():
    # 设置随机种子
    set_random_seed(seed=42)
    
    # 设置路径
    # input_dir = "/data/haoran/dataset/building3d/tokyo/training/xyz"  # 输入 .xyz 文件的目录
    # output_dir = "/data/haoran/dataset/building3d/tokyo/training_seg/xyz"  # 输出单体屋顶的目录
    input_dir = "/data/haoran/Point2Roof/tokyo_xyz"  # 输入 .xyz 文件的目录
    output_dir = "/data/haoran/Point2Roof/tokyo_xyz_res"  # 输出单体屋顶的目录
    # 参数设置（适合归一化点云）
    distance_threshold = 0.03  # 空间距离阈值0.02
    height_threshold = 0.010  # 高度差异阈值0.005 15
    min_points = 20  # 每个簇的最小点数
    point_threshold = 13000  # 点数阈值，超过此值才进行聚类
    
    process_xyz_files(input_dir, output_dir, distance_threshold, height_threshold, min_points, point_threshold)

if __name__ == "__main__":
    main()