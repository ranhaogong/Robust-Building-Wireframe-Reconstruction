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

def load_obj(file_path):
    """加载 .obj 文件，返回顶点和边"""
    vertices = []
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            elif line.startswith('l '):
                parts = line.strip().split()
                edge = [int(parts[1]) - 1, int(parts[2]) - 1]  # 转换为 0-based 索引
                edges.append(edge)
    return np.array(vertices), edges

def save_xyz(file_path, points, colors=None):
    """保存点云到 .xyz 文件，包含颜色信息（如果提供）"""
    if colors is not None:
        data = np.hstack((points, colors))
    else:
        data = points
    np.savetxt(file_path, data, fmt='%.6f')

def save_obj(file_path, vertices, edges):
    """保存 .obj 文件"""
    with open(file_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for e in edges:
            f.write(f"l {e[0] + 1} {e[1] + 1}\n")  # 转换为 1-based 索引

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

def enhanced_euclidean_clustering(pcd, sample_points_count, distance_threshold=0.03, height_threshold=0.05, is_training=True):
    """
    增强的欧几里得聚类，同时对样本点云和标签拐点进行聚类（训练集）或只对样本点云聚类（测试集）
    sample_points_count: 样本点云的点数，用于区分样本点和标签拐点
    is_training: 如果为 False，则忽略标签拐点（测试集）
    返回：(cluster_sample_points, cluster_vertex_points, cluster_sample_indices, cluster_vertex_indices)
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
        
        # 分离样本点和标签拐点
        cluster_points = points[current_cluster]
        sample_indices = [idx for idx in current_cluster if idx < sample_points_count]
        vertex_indices = [idx - sample_points_count for idx in current_cluster if idx >= sample_points_count]
        sample_points = cluster_points[[i for i, idx in enumerate(current_cluster) if idx < sample_points_count]]
        
        # 对于测试集，忽略标签拐点
        if is_training:
            vertex_points = cluster_points[[i for i, idx in enumerate(current_cluster) if idx >= sample_points_count]]
        else:
            vertex_points = np.array([])  # 测试集不处理标签拐点
        
        clusters.append((sample_points, vertex_points, sample_indices, vertex_indices))
    
    return clusters

def merge_small_clusters(clusters, min_points=20, is_training=True):
    """
    将点数小于 min_points 的簇中的每个点单独归并到最近的簇中
    clusters: (sample_points, vertex_points, sample_indices, vertex_indices)
    is_training: 如果为 False，则忽略标签拐点（测试集）
    """
    if not clusters:
        return clusters
    
    # 分离大簇和小簇（基于样本点数）
    large_clusters = [c for c in clusters if len(c[0]) >= min_points]
    small_clusters = [c for c in clusters if len(c[0]) < min_points]
    
    if not small_clusters or not large_clusters:
        return clusters
    
    # 为每个大簇构建 KD 树（仅使用样本点）
    large_cluster_kdtrees = []
    for sample_points, _, _, _ in large_clusters:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sample_points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        large_cluster_kdtrees.append(kdtree)
    
    # 存储合并后的簇
    merged_clusters = [[sp.copy(), vp.copy(), si.copy(), vi.copy()] for sp, vp, si, vi in large_clusters]
    
    # 对每个小簇中的每个点，找到最近的大簇
    for small_sample_points, small_vertex_points, small_sample_indices, small_vertex_indices in small_clusters:
        for point, sample_idx in zip(small_sample_points, small_sample_indices):
            min_dist = float('inf')
            nearest_cluster_idx = -1
            
            # 计算点到每个大簇中最近点的距离
            for i, kdtree in enumerate(large_cluster_kdtrees):
                [k, idx, dist] = kdtree.search_knn_vector_3d(point, 1)
                if dist[0] < min_dist:
                    min_dist = dist[0]
                    nearest_cluster_idx = i
            
            # 将样本点和索引添加到最近的大簇中
            merged_clusters[nearest_cluster_idx][0] = np.vstack([merged_clusters[nearest_cluster_idx][0], point])
            merged_clusters[nearest_cluster_idx][2].append(sample_idx)
        
        # 仅在训练集模式下处理标签拐点
        if is_training:
            for point, vertex_idx in zip(small_vertex_points, small_vertex_indices):
                min_dist = float('inf')
                nearest_cluster_idx = -1
                
                # 计算点到每个大簇中最近点的距离
                for i, kdtree in enumerate(large_cluster_kdtrees):
                    [k, idx, dist] = kdtree.search_knn_vector_3d(point, 1)
                    if dist[0] < min_dist:
                        min_dist = dist[0]
                        nearest_cluster_idx = i
                
                # 将标签拐点和索引添加到最近的大簇中
                merged_clusters[nearest_cluster_idx][1] = np.vstack([merged_clusters[nearest_cluster_idx][1], point]) if len(merged_clusters[nearest_cluster_idx][1]) > 0 else np.array([point])
                merged_clusters[nearest_cluster_idx][3].append(vertex_idx)
    
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

def process_xyz_files(input_xyz_dir, input_obj_dir, output_xyz_dir, output_obj_dir, distance_threshold=0.03, height_threshold=0.05, min_points=20, point_threshold=10000, is_training=True):
    """处理所有 .xyz 文件和 .obj 文件，同时进行聚类并保存分割结果
    is_training: 如果为 True，则同时分割数据和标签（训练集）；如果为 False，则只分割数据（测试集）
    """
    if not os.path.exists(output_xyz_dir):
        os.makedirs(output_xyz_dir)
    if is_training and not os.path.exists(output_obj_dir):
        os.makedirs(output_obj_dir)  # 仅在训练集模式下创建标签输出目录
    
    xyz_files = [f for f in os.listdir(input_xyz_dir) if f.endswith('.xyz')]
    
    for xyz_file in tqdm(xyz_files, desc="Processing files"):
        input_xyz_path = os.path.join(input_xyz_dir, xyz_file)
        base_name = os.path.splitext(xyz_file)[0]
        input_obj_path = os.path.join(input_obj_dir, f"{base_name}.obj")
        
        # 加载样本点云和标签拐点
        sample_points, input_colors = load_xyz(input_xyz_path)
        if is_training and not os.path.exists(input_obj_path):
            print(f"Warning: No corresponding .obj file found for {xyz_file}")
            vertices, edges = np.array([]), []
        elif is_training:
            vertices, edges = load_obj(input_obj_path)
        else:
            vertices, edges = np.array([]), []  # 测试集忽略标签
        
        # 检查总点数
        total_points = len(sample_points)
        if total_points <= point_threshold:
            # 点数不超过 threshold，直接保存原始文件
            output_xyz_file = os.path.join(output_xyz_dir, f"{base_name}_original.xyz")
            save_xyz(output_xyz_file, sample_points, input_colors)
            if is_training and os.path.exists(input_obj_path):
                output_obj_file = os.path.join(output_obj_dir, f"{base_name}_original.obj")
                shutil.copy(input_obj_path, output_obj_file)
            # print(f"Point count {total_points} <= {point_threshold}, saved original files")
            continue
        
        # 合并样本点云和标签拐点（训练集）
        if is_training and len(vertices) > 0:
            combined_points = np.vstack([sample_points, vertices])
        else:
            combined_points = sample_points  # 测试集只使用样本点云
        combined_points_normalized, center, scale = normalize_point_cloud(combined_points)
        
        # 转换为 open3d 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points_normalized)
        
        # 使用增强的欧几里得聚类
        initial_clusters = enhanced_euclidean_clustering(pcd, sample_points_count=len(sample_points), 
                                                        distance_threshold=distance_threshold, 
                                                        height_threshold=height_threshold,
                                                        is_training=is_training)
        
        if len(initial_clusters) == 0:
            print(f"Warning: No clusters found for {xyz_file}")
            continue
        
        # 合并小簇
        final_clusters = merge_small_clusters(initial_clusters, min_points=min_points, is_training=is_training)
        
        if len(final_clusters) == 0:
            print(f"Warning: No clusters found after merging for {xyz_file}")
            continue
        
        # 为可视化生成不同的颜色
        vis_colors = generate_distinct_colors(len(final_clusters))
        
        # 保存分割后的点云和标签
        for i, (cluster_sample_points, cluster_vertex_points, sample_indices, vertex_indices) in enumerate(final_clusters):
            # 处理样本点云
            cluster_sample_denormalized = denormalize_point_cloud(cluster_sample_points, center, scale)
            if input_colors is not None:
                cluster_colors = input_colors[sample_indices]
                cluster_with_colors = np.hstack((cluster_sample_denormalized, cluster_colors))
            else:
                cluster_colors = np.tile(vis_colors[i], (len(cluster_sample_denormalized), 1))
                cluster_with_colors = np.hstack((cluster_sample_denormalized, cluster_colors))
            
            output_xyz_file = os.path.join(output_xyz_dir, f"{base_name}_cluster_{i}.xyz")
            save_xyz(output_xyz_file, cluster_with_colors[:, :3], cluster_with_colors[:, 3:])
            # print(f"Saved cluster {i} with {len(cluster_sample_points)} sample points to {output_xyz_file}")
            
            # 处理标签拐点和边（仅训练集）
            if is_training and len(vertex_indices) > 0 and len(edges) > 0:
                cluster_vertices_denormalized = denormalize_point_cloud(cluster_vertex_points, center, scale)
                vertex_map = {idx: new_idx for new_idx, idx in enumerate(vertex_indices)}
                cluster_edges = [[vertex_map[edge[0]], vertex_map[edge[1]]] 
                               for edge in edges 
                               if edge[0] in vertex_map and edge[1] in vertex_map]
                
                output_obj_file = os.path.join(output_obj_dir, f"{base_name}_cluster_{i}.obj")
                save_obj(output_obj_file, cluster_vertices_denormalized, cluster_edges)
                # print(f"Saved cluster {i} with {len(cluster_vertex_points)} vertices to {output_obj_file}")
            # else:
                # print(f"Cluster {i} has no vertices or edges, skipping .obj file")

def main():
    # 设置随机种子
    set_random_seed(seed=42)
    
    # 设置路径
    # 示例 1：训练集（is_training=True）
    input_xyz_dir_train = "/data/haoran/dataset/building3d/tokyo/training/xyz"  # 输入 .xyz 文件的目录
    input_obj_dir_train = "/data/haoran/dataset/building3d/tokyo/training/wireframe"  # 输入 .obj 文件的目录
    output_xyz_dir_train = "/data/haoran/dataset/building3d/tokyo/training_seg/xyz"  # 输出分割点云的目录
    output_obj_dir_train = "/data/haoran/dataset/building3d/tokyo/training_seg/wireframe"  # 输出分割标签的目录
    
    # 示例 2：测试集（is_training=False）
    input_xyz_dir_test = "/data/haoran/Point2Roof/tokyo_xyz"  # 输入 .xyz 文件的目录
    input_obj_dir_test = "/data/haoran/dataset/building3d/tokyo/testing/wireframe"  # 输入 .obj 文件的目录（测试集不使用）
    output_xyz_dir_test = "/data/haoran/Point2Roof/tokyo_xyz_res"  # 输出分割点云的目录
    
    # 参数设置
    distance_threshold = 0.03  # 空间距离阈值
    height_threshold = 0.010  # 高度差异阈值
    min_points = 20  # 每个簇的最小点数
    point_threshold = 13000  # 点数阈值，超过此值才进行聚类
    
    # 处理训练集
    # print("Processing training set...")
    # process_xyz_files(input_xyz_dir_train, input_obj_dir_train, output_xyz_dir_train, output_obj_dir_train, 
    #                   distance_threshold, height_threshold, min_points, point_threshold, is_training=True)
    
    # 处理测试集
    print("Processing testing set...")
    process_xyz_files(input_xyz_dir_test, input_obj_dir_test, output_xyz_dir_test, None, 
                      distance_threshold, height_threshold, min_points, point_threshold, is_training=False)

if __name__ == "__main__":
    main()