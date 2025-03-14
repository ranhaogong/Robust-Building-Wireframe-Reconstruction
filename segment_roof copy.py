import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import random

def set_random_seed(seed=42):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    o3d.utility.random.seed(seed)  # 如果使用 open3d 的随机函数

def load_xyz(file_path):
    """加载 .xyz 文件"""
    points = np.loadtxt(file_path)[:, :3]  # 只取前三列 (x, y, z)
    return points

def save_xyz(file_path, points):
    """保存点云到 .xyz 文件"""
    np.savetxt(file_path, points, fmt='%.6f')

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

def euclidean_clustering(pcd, distance_threshold=0.03, min_cluster_size=50):
    """使用欧几里得聚类进行空间聚类"""
    labels = np.array(pcd.cluster_dbscan(eps=distance_threshold, min_points=min_cluster_size))
    clusters = []
    for label in np.unique(labels):
        if label >= 0:  # 忽略噪声点 (label = -1)
            cluster_points = np.asarray(pcd.points)[labels == label]
            clusters.append(cluster_points)
    return clusters

def generate_distinct_colors(n_clusters):
    """为每个簇生成不同的颜色（0-255 格式）"""
    hsv_colors = np.zeros((n_clusters, 3))
    hsv_colors[:, 0] = np.linspace(0, 360, n_clusters)
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


def process_xyz_files(input_dir, output_dir, distance_threshold=0.03, min_cluster_size=50):
    """处理所有 .xyz 文件并进行单体化"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    xyz_files = [f for f in os.listdir(input_dir) if f.endswith('.xyz')]
    
    for xyz_file in tqdm(xyz_files, desc="Processing files"):
        input_path = os.path.join(input_dir, xyz_file)
        points = load_xyz(input_path)
        points_normalized, center, scale = normalize_point_cloud(points)
        
        # 转换为 open3d 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_normalized)
        
        # 使用欧几里得聚类进行聚类
        all_clusters = euclidean_clustering(pcd, distance_threshold=distance_threshold, 
                                          min_cluster_size=min_cluster_size)
        
        if len(all_clusters) == 0:
            print(f"Warning: No clusters found for {xyz_file}")
            continue
        
        # 为每个簇生成不同的颜色
        colors = generate_distinct_colors(len(all_clusters))
        
        # 合并所有簇到一个点云中，并添加颜色
        merged_points_with_colors = []
        for i, cluster in enumerate(all_clusters):
            cluster_denormalized = denormalize_point_cloud(cluster, center, scale)
            cluster_colors = np.tile(colors[i], (len(cluster_denormalized), 1))
            cluster_with_colors = np.hstack((cluster_denormalized, cluster_colors))
            merged_points_with_colors.append(cluster_with_colors)
        
        merged_points_with_colors = np.vstack(merged_points_with_colors)

        
        # 保存到一个 .xyz 文件
        base_name = os.path.splitext(xyz_file)[0]
        output_file = os.path.join(output_dir, f"{base_name}_segmented.xyz")
        save_xyz(output_file, merged_points_with_colors)
        print(f"Saved segmented result to {output_file}")

        # 单独保存每个簇到单独的 .xyz 文件
        # for i, cluster in enumerate(all_clusters):
        #     cluster_denormalized = denormalize_point_cloud(cluster, center, scale)
        #     cluster_colors = np.tile(colors[i], (len(cluster_denormalized), 1))
        #     cluster_with_colors = np.hstack((cluster_denormalized, cluster_colors))
        #     output_file = os.path.join(output_dir, f"{base_name}_cluster_{i}.xyz")
        #     save_xyz(output_file, cluster_with_colors)
        #     print(f"Saved cluster {i} to {output_file}")

def main():
    # 设置随机种子
    set_random_seed(seed=42)
    # 设置路径
    input_dir = "/data/haoran/Point2Roof/tokyo_xyz"  # 输入 .xyz 文件的目录
    output_dir = "/data/haoran/Point2Roof/tokyo_xyz"  # 输出单体屋顶的目录
    
    # 参数设置（适合归一化点云）
    distance_threshold = 0.03  # 聚类距离阈值（控制相邻屋顶的分离）
    min_cluster_size = 30  # 最小簇大小
    
    process_xyz_files(input_dir, output_dir, distance_threshold, min_cluster_size)

if __name__ == "__main__":
    main()