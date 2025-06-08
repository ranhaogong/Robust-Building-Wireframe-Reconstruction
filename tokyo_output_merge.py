# import os
# from collections import defaultdict

# # 定义源目录和输出目录
# source_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_lovasz_edge/save"
# output_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_lovasz_edge/save_merge"

# # 确保输出目录存在
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 用于存储每个id的所有顶点和边的字典
# data_by_id = defaultdict(lambda: {'vertices': [], 'lines': []})
# original_files = {}  # 存储original文件内容

# # 第一步：读取所有文件并分类
# for filename in os.listdir(source_dir):
#     if not filename.endswith('.obj'):
#         continue
        
#     filepath = os.path.join(source_dir, filename)
    
#     # 解析文件名
#     parts = filename.split('_')
    
#     if 'cluster' in parts:
#         # 处理 cluster 文件
#         id_part = parts[1]  # tokyo_id_cluster_cid.obj 中的 id
        
#         # 读取文件内容
#         vertices = []
#         lines = []
#         with open(filepath, 'r') as f:
#             for line in f:
#                 if line.startswith('v '):
#                     vertices.append(line.strip())
#                 elif line.startswith('l '):
#                     lines.append(line.strip())
                    
#         # 存储到对应id的数据中
#         data_by_id[id_part]['vertices'].extend(vertices)
#         data_by_id[id_part]['lines'].append(lines)  # 按文件分开存储lines

#     else:
#         # 处理 original 文件
#         id_part = parts[1]  # tokyo_id_original.obj 中的 id
#         output_filename = f"tokyo_{id_part}.obj"
        
#         # 读取并存储文件内容
#         with open(filepath, 'r') as f:
#             original_files[id_part] = f.read()

# # 第二步：处理并输出文件
# # 先处理original文件
# for id_part, content in original_files.items():
#     output_filename = f"tokyo_{id_part}.obj"
#     output_path = os.path.join(output_dir, output_filename)
#     with open(output_path, 'w') as f:
#         f.write(content)

# # 再处理cluster文件
# for id_part, data in data_by_id.items():
#     output_filename = f"tokyo_{id_part}.obj"
#     output_path = os.path.join(output_dir, output_filename)
    
#     # 所有顶点按顺序排列，不去重
#     all_vertices = data['vertices']
#     new_lines = []
    
#     # 计算每个cluster文件的顶点偏移量并调整边的索引
#     vertex_offset = 0
#     for lines in data['lines']:  # 遍历每个cluster文件的边列表
#         for line in lines:
#             _, v1, v2 = line.split()
#             new_v1 = int(v1) + vertex_offset
#             new_v2 = int(v2) + vertex_offset
#             new_lines.append(f"l {new_v1} {new_v2}")
#         # 更新偏移量为当前cluster文件的顶点数量
#         vertex_offset += sum(1 for v in data['vertices'][:vertex_offset + len(lines)]) if vertex_offset == 0 else len(lines)
    
#     # 写入文件
#     with open(output_path, 'w') as f:
#         f.write('\n'.join(all_vertices) + '\n')
#         f.write('\n'.join(new_lines) + '\n')

# print(f"处理完成。保存了 {len(original_files)} 个original文件，合并了 {len(data_by_id)} 个ID的cluster文件。")


import os
from collections import defaultdict

# 定义源目录和输出目录
source_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_mrgd_lovasz_edge_dbscan_003/save"
output_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_mrgd_lovasz_edge_dbscan_003/save_merge_dbscan_0015_mrgd_edge_pred_07"
# source_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_cross_attention_lovasz_edge_finetune_dbscan_003/save"
# output_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_cross_attention_lovasz_edge_finetune_dbscan_003/save_merge_fpfh_cross_attention_dbscan_0015_edge_pred_06_epoch_264"
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 保存 original 文件内容，直接复制即可
original_files = {}

# 保存 cluster 文件：id -> list of (vertices, lines)
clusters_by_id = defaultdict(list)

# 遍历源目录
for filename in os.listdir(source_dir):
    if not filename.endswith('.obj'):
        continue

    parts = filename.split('_')

    if 'cluster' in parts:
        # cluster 文件，格式 tokyo_38_cluster_0.obj
        id_part = parts[1]  # e.g., '38'
        filepath = os.path.join(source_dir, filename)

        # 读取 cluster 文件内容
        vertices = []
        lines = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append(line.strip())
                elif line.startswith('l '):
                    lines.append(line.strip())

        clusters_by_id[id_part].append((vertices, lines))

    else:
        # original 文件，直接记录路径，不做处理
        id_part = parts[1]
        filepath = os.path.join(source_dir, filename)
        with open(filepath, 'r') as f:
            original_files[id_part] = f.read()

# Step 1: 写入 original 文件（不处理，直接复制）
for id_part, content in original_files.items():
    output_path = os.path.join(output_dir, f"tokyo_{id_part}.obj")
    with open(output_path, 'w') as f:
        f.write(content.strip() + '\n')

# Step 2: 合并并写入 cluster 文件
for id_part, cluster_list in clusters_by_id.items():
    output_path = os.path.join(output_dir, f"tokyo_{id_part}.obj")

    merged_vertices = []
    merged_lines = []
    vertex_offset = 0

    for vertices, lines in cluster_list:
        local_index_map = {}  # 当前 cluster 中的局部 index -> 全局 index

        # 为当前 cluster 的所有顶点分配全局编号
        for i, v in enumerate(vertices):
            global_index = vertex_offset + 1
            local_index_map[i + 1] = global_index
            merged_vertices.append(v)
            vertex_offset += 1

        # 更新 lines 中的索引
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0] == 'l':
                _, v1, v2 = parts
                new_v1 = local_index_map[int(v1)]
                new_v2 = local_index_map[int(v2)]
                merged_lines.append(f"l {new_v1} {new_v2}")

    # 写入合并后的 obj 文件
    with open(output_path, 'w') as f:
        f.write('\n'.join(merged_vertices) + '\n')
        f.write('\n'.join(merged_lines) + '\n')

print(f"✅ 处理完成：{len(original_files)} 个 original 文件，{len(clusters_by_id)} 个 cluster 文件被合并。")

