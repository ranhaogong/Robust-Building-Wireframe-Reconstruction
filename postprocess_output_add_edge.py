import numpy as np
import os
import glob
from scipy.spatial import KDTree
import math
from tqdm import tqdm # 导入 tqdm
from collections import defaultdict, deque # 用于图遍历

# --- 文件加载和保存辅助函数 ---

def load_xyz(filepath):
    """
    加载 .xyz 文件，返回点云数据 (Nx6)，其中前三列是XYZ坐标，后三列是RGB颜色。
    Args:
        filepath (str): .xyz 文件的完整路径。
    Returns:
        np.array: 加载的点云数据，形状为 (N, 6)。
    """
    try:
        return np.loadtxt(filepath)
    except Exception as e:
        # 使用 tqdm.write 来确保在进度条上方正确显示信息
        tqdm.write(f"错误: 无法加载 XYZ 文件 {filepath}: {e}")
        return None

def load_obj_wireframe(filepath):
    """
    加载 .obj 文件中的顶点和边信息。
    假设 obj 文件中，'v' 表示顶点，'l' 表示边（由两个顶点索引组成）。
    Args:
        filepath (str): .obj 文件的完整路径。
    Returns:
        tuple: (vertices, edges)
            vertices (np.array): Nx3 的顶点坐标数组。
            edges (np.array): Mx2 的边索引数组 (0-based)。
    """
    vertices = []
    edges = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    # 确保有足够的部件来解析XYZ坐标
                    if len(parts) >= 4:
                        vertices.append([float(p) for p in parts[1:4]])
                    else:
                        tqdm.write(f"警告: OBJ文件 {filepath} 中发现不完整的顶点行: {line.strip()}")
                elif parts[0] == 'l':
                    # OBJ 索引通常从 1 开始，需要转换为 0-based
                    # 确保有足够的部件来解析两个顶点索引
                    if len(parts) >= 3:
                        edges.append([int(p) - 1 for p in parts[1:3]])
                    else:
                        tqdm.write(f"警告: OBJ文件 {filepath} 中发现不完整的边行: {line.strip()}")
        return np.array(vertices), np.array(edges)
    except Exception as e:
        tqdm.write(f"错误: 无法加载 OBJ 文件 {filepath}: {e}")
        return None, None

def save_obj_wireframe(filepath, vertices, edges):
    """
    将顶点和边保存为 .obj 文件。
    Args:
        filepath (str): 保存 .obj 文件的完整路径。
        vertices (np.array): Nx3 的顶点坐标数组。
        edges (np.array): Mx2 的边索引数组 (0-based)。
    """
    try:
        with open(filepath, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for e in edges:
                # OBJ 索引通常从 1 开始
                f.write(f"l {e[0]+1} {e[1]+1}\n")
    except Exception as e:
        tqdm.write(f"错误: 无法保存 OBJ 文件 {filepath}: {e}")

# --- 点云处理和归一化函数 ---

def normalize_points(points):
    """
    根据用户提供的逻辑归一化点云。
    将点云缩放到以原点为中心，最大距离为 1 的单位球内。
    Args:
        points (np.array): Nx3 的点坐标数组。
    Returns:
        tuple: (normalized_points, centroid, max_distance)
            normalized_points (np.array): 归一化后的点坐标。
            centroid (np.array): 用于归一化的中心点。
            max_distance (float): 用于归一化的最大距离。
    """
    if points.shape[0] == 0:
        return np.array([]), np.array([0,0,0]), 1.0 # 处理空点云
        
    min_pt, max_pt = np.min(points, axis=0), np.max(points, axis=0)
    maxXYZ = np.max(max_pt)
    minXYZ = np.min(min_pt)
    
    centroid = np.mean(points, axis=0)
    normalized_points = points - centroid
    
    distances = np.linalg.norm(normalized_points, axis=1)
    max_distance = np.max(distances)
    
    if max_distance == 0: # 避免除以零，例如当所有点都相同时
        return normalized_points, centroid, 1.0 # 返回未缩放但已中心化的点，max_distance设为1
    
    normalized_points /= max_distance
    return normalized_points, centroid, max_distance

# --- 法线估计函数 ---

def estimate_normals(points, k_neighbors=10):
    """
    使用 PCA 为点云中的每个点估计法线。
    Args:
        points (np.array): Nx3 的点坐标数组 (应为归一化后的点云)。
        k_neighbors (int): 用于 PCA 的邻居点数量。
    Returns:
        np.array: Nx3 的估计法线数组。
    """
    if points.shape[0] == 0:
        return np.array([])

    kd_tree = KDTree(points)
    normals = np.zeros_like(points)

    for i, p in enumerate(points):
        # 查询 k+1 个邻居，因为第一个是点本身
        distances, indices = kd_tree.query(p, k=k_neighbors + 1)
        neighbors = points[indices[1:]] # 排除点本身

        if len(neighbors) < 3: # 至少需要 3 个点才能进行 PCA
            # 如果邻居不足，可以设置一个默认法线或跳过
            normals[i] = np.array([0.0, 0.0, 1.0]) # 默认向上
            continue

        # 将邻居点中心化
        centered_neighbors = neighbors - np.mean(neighbors, axis=0)
        
        # 计算协方差矩阵
        # rowvar=False 表示每列是一个变量，每行是一个观测
        covariance_matrix = np.cov(centered_neighbors, rowvar=False)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # 最小特征值对应的特征向量就是法线方向
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        
        # 确保法线方向一致性（例如，Z分量始终为正，表示“向上”）
        # 这对于后续的法线一致性检查很重要
        # 更好的方法是使用一致的方向（例如，朝向点云的质心或远离）
        # 但对于屋顶，向上通常是合理的
        if normal[2] < 0: # 假设 Z 轴是“向上”方向
            normal = -normal
        
        normals[i] = normal / np.linalg.norm(normal) # 归一化为单位向量
    return normals

# --- 后处理主函数 ---

def post_process_wireframe(vertices, edges, original_point_cloud,
                           min_edge_length_threshold=0.005, # 归一化空间中的阈值，默认值更宽松
                           min_points_on_edge=2, # 默认值更宽松
                           normal_consistency_threshold=math.cos(math.radians(40)), # 默认值更宽松 (40度夹角)
                           max_gap_fill_distance_multiplier=5.0, # 闭合平面时，最大连接距离倍数
                           normal_consistency_threshold_for_gaps=math.cos(math.radians(50)), # 闭合平面时，法线一致性阈值更宽松
                           min_points_on_gap_edge=1, # 闭合平面时，新边的点云支撑点数要求更宽松
                           min_angle_for_gap_fill=math.radians(10), # 闭合平面时，避免锐角，弧度
                           ):
    """
    对 wireframe 进行后处理，包括短边过滤、点云支撑度检查、法线一致性检查和闭合平面。
    所有空间相关的参数（如长度、半径）都应在归一化空间中考虑。
    Args:
        vertices (np.array): Nx3 的顶点坐标数组 (未归一化)。
        edges (np.array): Mx2 的边索引数组。
        original_point_cloud (np.array): Kx6 的原始点云数据 (未归一化)。
        min_edge_length_threshold (float): 短边长度阈值，小于此值的边将被删除 (在归一化空间中)。
        min_points_on_edge (int): 一条边上至少有多少个采样点，其周围有足够点云支撑才保留该边。
        normal_consistency_threshold (float): 法线一致性阈值 (cos值)，如果两个顶点法线点积的绝对值小于此值，则删除边。
        max_gap_fill_distance_multiplier (float): 闭合平面时，允许连接的两个点之间的最大距离倍数。
        normal_consistency_threshold_for_gaps (float): 闭合平面时，法线一致性阈值。
        min_points_on_gap_edge (int): 闭合平面时，新边的点云支撑点数要求。
        min_angle_for_gap_fill (float): 闭合平面时，限制新连接的边与现有边形成的最小夹角（弧度）。
    Returns:
        tuple: (processed_vertices, processed_edges)
    """
    
    # 1. 归一化原始点云
    original_xyz = original_point_cloud[:, :3]
    normalized_point_cloud_xyz, centroid, max_distance = normalize_points(original_xyz.copy())
    
    # --- 自适应参数计算 ---
    # 自适应 k_neighbors_for_normals (法线估计的邻居数)
    # 确保 k_neighbors 至少为 3 (PCA 要求)，且不超过点云总数的某个比例或最大值
    if original_point_cloud.shape[0] > 0:
        adaptive_k_neighbors_for_normals = max(10, min(int(original_point_cloud.shape[0] * 0.01), 50))
    else:
        adaptive_k_neighbors_for_normals = 10 # 默认值
    tqdm.write(f"  自适应法线估计邻居数 k 设置为: {adaptive_k_neighbors_for_normals}")

    # 2. 估计归一化点云的法线
    tqdm.write(f"  估计点云法线 (k={adaptive_k_neighbors_for_normals})...")
    point_cloud_normals = estimate_normals(normalized_point_cloud_xyz, k_neighbors=adaptive_k_neighbors_for_normals)
    # 为法线查询构建 KDTree，使用归一化点云的坐标
    kd_tree_for_normals_query = KDTree(normalized_point_cloud_xyz) 
    tqdm.write("  法线估计完成。")

    # 自适应 point_cloud_support_radius (点云支撑半径)
    # 基于归一化点云中 k-NN 的平均距离来估计局部密度
    adaptive_point_cloud_support_radius = 0.02 # 默认值
    if normalized_point_cloud_xyz.shape[0] > adaptive_k_neighbors_for_normals:
        # 采样一部分点来计算平均邻居距离，避免对所有点计算，提高效率
        sample_indices = np.random.choice(normalized_point_cloud_xyz.shape[0], 
                                          min(1000, normalized_point_cloud_xyz.shape[0]), 
                                          replace=False)
        sample_points = normalized_point_cloud_xyz[sample_indices]
        
        distances, _ = kd_tree_for_normals_query.query(sample_points, k=adaptive_k_neighbors_for_normals + 1)
        # 排除自身距离 (第0列)，取剩余邻居距离的平均值
        avg_nn_distance = np.mean(distances[:, 1:]) 
        
        # 将支撑半径设置为平均邻居距离的某个倍数 (例如 1.5 到 3.0 倍)
        # 确保它在一个合理的范围内，避免过小或过大
        adaptive_point_cloud_support_radius = max(0.01, min(0.05, avg_nn_distance * 2.0)) 
        tqdm.write(f"  自适应点云支撑半径设置为: {adaptive_point_cloud_support_radius:.4f}")
    else:
        tqdm.write("  点云数量不足，无法自适应点云支撑半径，使用默认值。")

    # 3. 归一化 wireframe 顶点，以便与归一化点云进行比较
    # 确保 max_distance 不为零，避免除以零错误
    if max_distance == 0:
        tqdm.write("  警告: 点云最大距离为零，无法归一化 wireframe 顶点。跳过归一化。")
        normalized_vertices = vertices - centroid # 仅中心化
    else:
        normalized_vertices = (vertices - centroid) / max_distance
    
    current_edges = list(edges) # 将 NumPy 数组转换为列表以便修改
    
    tqdm.write(f"  原始边数量: {len(current_edges)}")
    
    # 过滤步骤 1: 短边过滤
    filtered_edges_by_length = []
    for v1_idx, v2_idx in current_edges:
        if v1_idx >= len(normalized_vertices) or v2_idx >= len(normalized_vertices):
            tqdm.write(f"  警告: 边 ({v1_idx}, {v2_idx}) 包含超出范围的顶点索引。跳过此边。")
            continue
        p1 = normalized_vertices[v1_idx]
        p2 = normalized_vertices[v2_idx]
        length = np.linalg.norm(p1 - p2)
        if length >= min_edge_length_threshold:
            filtered_edges_by_length.append((v1_idx, v2_idx))
    tqdm.write(f"  短边过滤后剩余边数量: {len(filtered_edges_by_length)}")

    # 过滤步骤 2: 基于点云支撑度检查
    filtered_edges_by_support = []
    for v1_idx, v2_idx in filtered_edges_by_length:
        p1 = normalized_vertices[v1_idx]
        p2 = normalized_vertices[v2_idx]
        
        num_samples = 10 # 在边上采样的点数
        t_values = np.linspace(0, 1, num_samples)
        # 确保 p1 和 p2 是 NumPy 数组以便进行广播操作
        sampled_points_on_edge = p1 + t_values[:, np.newaxis] * (p2 - p1)

        supported_sample_points_count = 0
        for sp in sampled_points_on_edge:
            # 在原始点云中搜索采样点附近的点
            indices = kd_tree_for_normals_query.query_ball_point(sp, adaptive_point_cloud_support_radius)
            if len(indices) > 0: # 如果找到了点，认为这个采样点有支撑
                supported_sample_points_count += 1
        
        # 如果边上足够多的采样点有点云支撑，则保留这条边
        if supported_sample_points_count >= min_points_on_edge:
            filtered_edges_by_support.append((v1_idx, v2_idx))
    
    tqdm.write(f"  点云支撑度过滤后剩余边数量: {len(filtered_edges_by_support)}")

    # 过滤步骤 3: 法线一致性检查
    processed_edges = []
    for v1_idx, v2_idx in filtered_edges_by_support:
        p1_norm = normalized_vertices[v1_idx]
        p2_norm = normalized_vertices[v2_idx]

        # 找到 wireframe 顶点在归一化点云中最近的点，并获取其估计的法线
        # query 返回 (距离, 索引)
        _, idx1 = kd_tree_for_normals_query.query(p1_norm)
        _, idx2 = kd_tree_for_normals_query.query(p2_norm)

        # 确保索引在有效范围内
        if idx1 >= len(point_cloud_normals) or idx2 >= len(point_cloud_normals):
            tqdm.write(f"  警告: 法线查询索引超出范围。跳过边 ({v1_idx}, {v2_idx})。")
            continue

        normal1 = point_cloud_normals[idx1]
        normal2 = point_cloud_normals[idx2]

        # 计算法线点积的绝对值
        dot_product = np.abs(np.dot(normal1, normal2))
        
        # 如果法线一致性低于阈值（即夹角过大），则删除这条边
        if dot_product >= normal_consistency_threshold:
            processed_edges.append((v1_idx, v2_idx))
        # else:
            # tqdm.write(f"  因法线不一致移除边 ({v1_idx}, {v2_idx})。点积: {dot_product:.3f}")
    
    tqdm.write(f"  法线一致性过滤后剩余边数量: {len(processed_edges)}")

    # --- 闭合平面 (Targeted Gap Closing) 步骤 ---
    tqdm.write("  开始目标性闭合平面...")
    newly_added_edges = []
    
    # 1. 构建图的邻接列表和计算度
    adj_list = defaultdict(list)
    degrees = defaultdict(int)
    for u, v in processed_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        degrees[u] += 1
        degrees[v] += 1

    # 2. 查找连通分量及其开放端点
    visited = set()
    components_to_close = [] # 存储待闭合的连通分量的 (端点1索引, 端点2索引)

    for start_node in range(len(normalized_vertices)): # 遍历所有可能的顶点
        if start_node not in visited and degrees[start_node] > 0: # 只考虑有边的顶点
            current_component_nodes = []
            current_component_endpoints = []
            
            queue = deque([start_node]) # 使用 deque 进行 BFS
            visited.add(start_node)

            while queue:
                u = queue.popleft()
                current_component_nodes.append(u)
                if degrees[u] == 1: # 识别度为 1 的顶点作为开放端点
                    current_component_endpoints.append(u)
                
                for v in adj_list[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            
            # 如果一个连通分量恰好有两个开放端点，则将其加入待闭合列表
            if len(current_component_endpoints) == 2:
                components_to_close.append(tuple(current_component_endpoints))
            # else:
                # tqdm.write(f"  跳过连通分量 (节点数: {len(current_component_nodes)}, 端点数: {len(current_component_endpoints)})，不符合闭合条件。")

    tqdm.write(f"  找到 {len(components_to_close)} 个待闭合的连通分量。")

    # 3. 尝试连接这些开放端点
    existing_edges_set = set() # 重新构建集合，用于快速查找是否已存在
    for u, v in processed_edges:
        existing_edges_set.add(tuple(sorted((u, v))))

    for ep1_idx, ep2_idx in components_to_close:
        # 如果这两个顶点已经连接，则跳过 (理论上不应发生，因为它们是度为1的端点)
        if tuple(sorted((ep1_idx, ep2_idx))) in existing_edges_set:
            continue

        p1_norm = normalized_vertices[ep1_idx]
        p2_norm = normalized_vertices[ep2_idx]
        
        # 距离检查
        # dist = np.linalg.norm(p1_norm - p2_norm)
        # if dist > adaptive_point_cloud_support_radius * max_gap_fill_distance_multiplier:
        #     # tqdm.write(f"    跳过连接 ({ep1_idx}, {ep2_idx})：距离太远 ({dist:.4f} > {adaptive_point_cloud_support_radius * max_gap_fill_distance_multiplier:.4f})")
        #     continue

        # 法线一致性检查 (针对闭合平面，可能更宽松)
        _, idx1 = kd_tree_for_normals_query.query(p1_norm)
        _, idx2 = kd_tree_for_normals_query.query(p2_norm)

        if idx1 >= len(point_cloud_normals) or idx2 >= len(point_cloud_normals):
            # tqdm.write(f"    跳过连接 ({ep1_idx}, {ep2_idx})：法线查询索引超出范围。")
            continue

        normal1 = point_cloud_normals[idx1]
        normal2 = point_cloud_normals[idx2]
        dot_product = np.abs(np.dot(normal1, normal2))
        if dot_product < normal_consistency_threshold_for_gaps:
            # tqdm.write(f"    跳过连接 ({ep1_idx}, {ep2_idx})：法线不一致 (点积: {dot_product:.3f} < {normal_consistency_threshold_for_gaps:.3f})")
            continue

        # 点云支撑度检查 (针对闭合平面，可能更宽松)
        num_samples = 10 
        t_values = np.linspace(0, 1, num_samples)
        sampled_points_on_candidate_edge = p1_norm + t_values[:, np.newaxis] * (p2_norm - p1_norm)
        
        supported_sample_points_count = 0
        for sp in sampled_points_on_candidate_edge:
            indices = kd_tree_for_normals_query.query_ball_point(sp, adaptive_point_cloud_support_radius)
            if len(indices) > 0:
                supported_sample_points_count += 1
        
        if supported_sample_points_count < min_points_on_gap_edge:
            # tqdm.write(f"    跳过连接 ({ep1_idx}, {ep2_idx})：点云支撑不足 ({supported_sample_points_count} < {min_points_on_gap_edge})")
            continue

        # 角度检查 (避免添加不合理的锐角或接近180度的角)
        is_angle_valid = True
        # 检查新边与连接到 ep1_idx 的现有边之间的夹角
        for neighbor_idx in adj_list[ep1_idx]:
            vec1 = normalized_vertices[neighbor_idx] - p1_norm
            vec2 = p2_norm - p1_norm # 候选边向量
            if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
                dot_prod_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                # 限制 dot_prod_angle 在 [-1, 1] 之间，避免浮点误差导致 math.acos 错误
                dot_prod_angle = np.clip(dot_prod_angle, -1.0, 1.0)
                angle = math.acos(dot_prod_angle)
                if angle < min_angle_for_gap_fill or angle > (math.pi - min_angle_for_gap_fill): # 避免锐角或接近180度的角
                    is_angle_valid = False
                    # tqdm.write(f"    跳过连接 ({ep1_idx}, {ep2_idx})：与现有边 ({ep1_idx}-{neighbor_idx}) 形成不合理角度 ({math.degrees(angle):.2f}度)")
                    break
        if not is_angle_valid:
            continue

        # 检查新边与连接到 ep2_idx 的现有边之间的夹角
        for neighbor_idx in adj_list[ep2_idx]:
            vec1 = normalized_vertices[neighbor_idx] - p2_norm
            vec2 = p1_norm - p2_norm # 候选边向量
            if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
                dot_prod_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                dot_prod_angle = np.clip(dot_prod_angle, -1.0, 1.0)
                angle = math.acos(dot_prod_angle)
                if angle < min_angle_for_gap_fill or angle > (math.pi - min_angle_for_gap_fill):
                    is_angle_valid = False
                    # tqdm.write(f"    跳过连接 ({ep1_idx}, {ep2_idx})：与现有边 ({ep2_idx}-{neighbor_idx}) 形成不合理角度 ({math.degrees(angle):.2f}度)")
                    break
        if not is_angle_valid:
            continue

        # 如果通过所有检查，则添加这条边
        newly_added_edges.append((ep1_idx, ep2_idx))
        existing_edges_set.add(tuple(sorted((ep1_idx, ep2_idx)))) # 更新集合，避免重复添加
        # tqdm.write(f"    成功添加边: ({ep1_idx}, {ep2_idx})")
    
    processed_edges.extend(newly_added_edges)
    tqdm.write(f"  闭合平面后新增边数量: {len(newly_added_edges)}")
    tqdm.write(f"  最终边数量: {len(processed_edges)}")

    # 返回原始顶点，因为我们只修改了边。
    # 如果需要删除孤立顶点，则需要额外的处理来重新映射索引。
    return vertices, np.array(processed_edges)

# --- 主执行逻辑 ---

if __name__ == "__main__":
    # 配置输入和输出目录
    input_obj_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_mrgd_lovasz_edge_dbscan_003/save_merge_dbscan_0015_mrgd_pred_logit_05"
    input_xyz_dir = "/data/haoran/dataset/building3d/tokyo/testing/xyz"
    output_obj_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_mrgd_lovasz_edge_dbscan_003/save_merge_dbscan_0015_mrgd_pred_logit_05_refine_add_edge"

    # 创建输出目录，如果它不存在
    os.makedirs(output_obj_dir, exist_ok=True)
    print(f"输出目录: {output_obj_dir}")

    # 获取所有 .obj 文件
    obj_files = glob.glob(os.path.join(input_obj_dir, "*.obj"))
    if not obj_files:
        print(f"在 {input_obj_dir} 中未找到任何 .obj 文件。请检查路径。")
    
    total_processed_files = 0

    # 使用 tqdm 包装文件处理循环以显示进度条
    for obj_file_path in tqdm(obj_files, desc="Processing files"):
        filename_without_ext = os.path.splitext(os.path.basename(obj_file_path))[0]
        xyz_file_path = os.path.join(input_xyz_dir, f"{filename_without_ext}.xyz")
        output_file_path = os.path.join(output_obj_dir, f"{filename_without_ext}.obj")

        # 使用 tqdm.write 来确保在进度条上方正确显示文件处理信息
        tqdm.write(f"\n处理文件: {filename_without_ext}")

        # 1. 加载数据
        point_cloud_data = load_xyz(xyz_file_path)
        if point_cloud_data is None or point_cloud_data.shape[0] == 0:
            tqdm.write(f"  跳过文件 {filename_without_ext}，因为 XYZ 数据加载失败或为空。")
            continue

        predicted_vertices, predicted_edges = load_obj_wireframe(obj_file_path)
        if predicted_vertices is None or predicted_edges is None:
            tqdm.write(f"  跳过文件 {filename_without_ext}，因为 OBJ 数据加载失败。")
            continue
        
        if predicted_vertices.shape[0] == 0 or predicted_edges.shape[0] == 0:
            tqdm.write(f"  跳过文件 {filename_without_ext}，因为预测的 wireframe 顶点或边为空。")
            # 即使为空，也可以保存一个空的OBJ文件，或者选择跳过
            save_obj_wireframe(output_file_path, predicted_vertices, predicted_edges)
            total_processed_files += 1
            continue

        tqdm.write(f"  加载完成，顶点数量: {len(predicted_vertices)}, 边数量: {len(predicted_edges)}")

        # 2. 进行后处理
        tqdm.write("  开始进行后处理...")
        # 这些参数现在有更宽松的默认值，并且部分参数会根据点云密度自适应
        processed_vertices, processed_edges = post_process_wireframe(
            predicted_vertices, 
            predicted_edges, 
            point_cloud_data,
            min_edge_length_threshold=0.005, # 默认值更宽松 (0.5% of normalized extent)
            min_points_on_edge=2, # 默认值更宽松 (2 out of 10 sampled points needed)
            normal_consistency_threshold=math.cos(math.radians(40)), # 默认值更宽松 (允许40度夹角)
            max_gap_fill_distance_multiplier=5.0, # 闭合平面时，最大连接距离倍数
            normal_consistency_threshold_for_gaps=math.cos(math.radians(50)), # 闭合平面时，法线一致性阈值更宽松
            min_points_on_gap_edge=1, # 闭合平面时，新边的点云支撑点数要求更宽松
            min_angle_for_gap_fill=math.radians(10), # 闭合平面时，限制新连接的边与现有边形成的最小夹角（弧度）
        )

        # 3. 保存后处理结果
        save_obj_wireframe(output_file_path, processed_vertices, processed_edges)
        tqdm.write(f"  后处理结果已保存到: {output_file_path}")
        tqdm.write(f"  后处理后顶点数量: {len(processed_vertices)}, 边数量: {len(processed_edges)}")
        total_processed_files += 1
    
    print(f"\n所有文件处理完成。总共处理了 {total_processed_files} 个文件。")

