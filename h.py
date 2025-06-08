# import os
# import matplotlib.pyplot as plt
# from tqdm import tqdm  # 导入 tqdm 用于进度条显示

# # 文件夹路径
# xyz_folder = "/data/haoran/dataset/building3d/roof/Tallinn/train/xyz"
# obj_folder = "/data/haoran/dataset/building3d/roof/Tallinn/train/wireframe"

# # 存储每个文件的点数和边数
# point_counts = []
# edge_counts = []

# # 获取所有 .xyz 文件列表
# xyz_files = [f for f in os.listdir(xyz_folder) if f.endswith('.xyz')]

# # 使用 tqdm 包裹文件遍历，以显示进度条
# for xyz_file in tqdm(xyz_files, desc="Processing xyz files", unit="file"):
#     # 获取对应的 obj 文件路径
#     obj_file = xyz_file.replace('.xyz', '.obj')
    
#     # 统计 xyz 文件中的点数
#     with open(os.path.join(xyz_folder, xyz_file), 'r') as f:
#         points = [line for line in f]  # 每行有三个数字，表示一个点
#         point_counts.append(len(points))
    
#     # 统计 obj 文件中的边数
#     with open(os.path.join(obj_folder, obj_file), 'r') as f:
#         edges = [line for line in f if line.startswith('l')]  # 每行以 l 开头，表示一个边
#         edge_counts.append(len(edges))

# # 绘制点数的直方图
# plt.figure(figsize=(10, 6))
# plt.hist(point_counts, bins=20, color='b', alpha=0.7)
# plt.xlabel('Number of Points', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.title('Distribution of Point Counts per Sample', fontsize=14)
# point_hist_path = "point_count_histogram.png"
# plt.tight_layout()
# plt.savefig(point_hist_path)
# plt.show()

# # 绘制边数的直方图
# plt.figure(figsize=(10, 6))
# plt.hist(edge_counts, bins=20, color='g', alpha=0.7)
# plt.xlabel('Number of Edges', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.title('Distribution of Edge Counts per Sample', fontsize=14)
# edge_hist_path = "edge_count_histogram.png"
# plt.tight_layout()
# plt.savefig(edge_hist_path)
# plt.show()

# # 输出保存路径
# print(f"点数直方图已保存为 {point_hist_path}")
# print(f"边数直方图已保存为 {edge_hist_path}")


# import os

# # 定义源目录和目标文件路径
# source_dir = "/data/haoran/dataset/building3d/tokyo/testing_seg/xyz"
# output_file = "/data/haoran/dataset/building3d/Point2Roof_tokyo_seg/test_all.txt"

# # 确保输出目录存在
# output_dir = os.path.dirname(output_file)
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 获取所有.xyz文件的绝对路径
# xyz_files = []
# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         if file.endswith('.xyz'):
#             abs_path = os.path.abspath(os.path.join(root, file))
#             xyz_files.append(abs_path)

# # 将路径写入文件
# with open(output_file, 'w') as f:
#     for path in xyz_files:
#         f.write(f"{path}\n")

# print(f"已将 {len(xyz_files)} 个.xyz文件的绝对路径写入到 {output_file}")

# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm

# # 文件夹路径
# tallinn_folder = "/data/haoran/dataset/building3d/roof/Tallinn/train/xyz"
# second_folder = "/data/haoran/dataset/building3d/roof/Entry-level/train/xyz"  # 替换为第二个数据集路径

# # 存储点数
# tallinn_points = []
# second_points = []

# # 处理 Tallinn 数据集
# tallinn_files = [f for f in os.listdir(tallinn_folder) if f.endswith('.xyz')]
# for xyz_file in tqdm(tallinn_files, desc="Processing Tallinn xyz files", unit="file"):
#     with open(os.path.join(tallinn_folder, xyz_file), 'r') as f:
#         points = [line for line in f]
#         tallinn_points.append(len(points))

# # 处理第二个数据集
# second_files = [f for f in os.listdir(second_folder) if f.endswith('.xyz')]
# for xyz_file in tqdm(second_files, desc="Processing Entry-level xyz files", unit="file"):
#     with open(os.path.join(second_folder, xyz_file), 'r') as f:
#         points = [line for line in f]
#         second_points.append(len(points))

# # 设置 Times New Roman 字体
# plt.rcParams['font.family'] = 'Times New Roman'

# # 创建子图，左右并排
# fig, ax = plt.subplots(1, 2, figsize=(8, 5), dpi=100, sharey=True)

# # 绘制 Tallinn 小提琴图
# sns.violinplot(data=tallinn_points, color='#B36A6F', inner='quartile', linewidth=1.5, alpha=0.8, ax=ax[0])
# ax[0].set_xlabel('Points (Tallinn)', fontsize=12, labelpad=10)
# ax[0].set_ylabel('Density', fontsize=12, labelpad=10)
# ax[0].grid(True, linestyle='--', alpha=0.3)
# ax[0].spines['top'].set_visible(False)
# ax[0].spines['right'].set_visible(False)
# ax[0].spines['left'].set_linewidth(0.5)
# ax[0].spines['bottom'].set_linewidth(0.5)
# ax[0].tick_params(axis='both', which='major', labelsize=10)

# # 绘制第二个数据集小提琴图
# sns.violinplot(data=second_points, color='#4A7A96', inner='quartile', linewidth=1.5, alpha=0.8, ax=ax[1])
# ax[1].set_xlabel('Points (Entry-level)', fontsize=12, labelpad=10)
# ax[1].grid(True, linestyle='--', alpha=0.3)
# ax[1].spines['top'].set_visible(False)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['left'].set_linewidth(0.5)
# ax[1].spines['bottom'].set_linewidth(0.5)
# ax[1].tick_params(axis='both', which='major', labelsize=10)

# # 统一标题
# fig.suptitle('Point Count Distribution Across Datasets', fontsize=14, y=1.05)

# # 保存图像
# violin_path = "point_count_violin_dual.png"
# plt.tight_layout()
# plt.savefig(violin_path, dpi=300, bbox_inches='tight')
# plt.show()

# print(f"点数小提琴图已保存为 {violin_path}")

import os

# 定义输入和输出路径
input_dir = "/data/haoran/dataset/building3d/tokyo/training_seg_9000/xyz"
output_file = "/data/haoran/dataset/building3d/tokyo/training_seg_9000/xyz/test_all.txt"

# 确保输入目录存在
if not os.path.exists(input_dir):
    print(f"输入目录 {input_dir} 不存在")
    exit(1)

# 获取所有.xyz文件的绝对路径
xyz_files = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".xyz"):
            abs_path = os.path.abspath(os.path.join(root, file))
            xyz_files.append(abs_path)

# 将路径写入输出文件
with open(output_file, 'w') as f:
    for path in xyz_files:
        f.write(path + '\n')

print(f"已将 {len(xyz_files)} 个.xyz文件的绝对路径写入 {output_file}")