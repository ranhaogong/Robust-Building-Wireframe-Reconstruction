import os
from collections import defaultdict

# 定义源目录和输出目录
source_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_lovasz_edge/save"
output_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_lovasz_edge/save_merge"

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 用于存储每个id的所有顶点和边的字典
data_by_id = defaultdict(lambda: {'vertices': [], 'lines': []})
original_files = {}  # 存储original文件内容

# 第一步：读取所有文件并分类
for filename in os.listdir(source_dir):
    if not filename.endswith('.obj'):
        continue
        
    filepath = os.path.join(source_dir, filename)
    
    # 解析文件名
    parts = filename.split('_')
    
    if 'cluster' in parts:
        # 处理 cluster 文件
        id_part = parts[1]  # tokyo_id_cluster_cid.obj 中的 id
        
        # 读取文件内容
        vertices = []
        lines = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append(line.strip())
                elif line.startswith('l '):
                    lines.append(line.strip())
                    
        # 存储到对应id的数据中
        data_by_id[id_part]['vertices'].extend(vertices)
        data_by_id[id_part]['lines'].append(lines)  # 按文件分开存储lines

    else:
        # 处理 original 文件
        id_part = parts[1]  # tokyo_id_original.obj 中的 id
        output_filename = f"tokyo_{id_part}.obj"
        
        # 读取并存储文件内容
        with open(filepath, 'r') as f:
            original_files[id_part] = f.read()

# 第二步：处理并输出文件
# 先处理original文件
for id_part, content in original_files.items():
    output_filename = f"tokyo_{id_part}.obj"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write(content)

# 再处理cluster文件
for id_part, data in data_by_id.items():
    output_filename = f"tokyo_{id_part}.obj"
    output_path = os.path.join(output_dir, output_filename)
    
    # 所有顶点按顺序排列，不去重
    all_vertices = data['vertices']
    new_lines = []
    
    # 计算每个cluster文件的顶点偏移量并调整边的索引
    vertex_offset = 0
    for lines in data['lines']:  # 遍历每个cluster文件的边列表
        for line in lines:
            _, v1, v2 = line.split()
            new_v1 = int(v1) + vertex_offset
            new_v2 = int(v2) + vertex_offset
            new_lines.append(f"l {new_v1} {new_v2}")
        # 更新偏移量为当前cluster文件的顶点数量
        vertex_offset += sum(1 for v in data['vertices'][:vertex_offset + len(lines)]) if vertex_offset == 0 else len(lines)
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write('\n'.join(all_vertices) + '\n')
        f.write('\n'.join(new_lines) + '\n')

print(f"处理完成。保存了 {len(original_files)} 个original文件，合并了 {len(data_by_id)} 个ID的cluster文件。")

# import os

# # 定义目标目录
# target_dir = "/data/haoran/dataset/building3d/tokyo/testing"
# # target_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_lovasz_edge/save_merge"
# # /data/haoran/dataset/building3d/tokyo/testing
# # 统计文件数量
# file_count = 0
# for root, dirs, files in os.walk(target_dir):
#     file_count += len(files)

# print(f"文件夹 {target_dir} 中共有 {file_count} 个文件")