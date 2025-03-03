import os
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入 tqdm 用于进度条显示

# 文件夹路径
xyz_folder = "/data/haoran/dataset/building3d/roof/Tallinn/train/xyz"
obj_folder = "/data/haoran/dataset/building3d/roof/Tallinn/train/wireframe"

# 存储每个文件的点数和边数
point_counts = []
edge_counts = []

# 获取所有 .xyz 文件列表
xyz_files = [f for f in os.listdir(xyz_folder) if f.endswith('.xyz')]

# 使用 tqdm 包裹文件遍历，以显示进度条
for xyz_file in tqdm(xyz_files, desc="Processing xyz files", unit="file"):
    # 获取对应的 obj 文件路径
    obj_file = xyz_file.replace('.xyz', '.obj')
    
    # 统计 xyz 文件中的点数
    with open(os.path.join(xyz_folder, xyz_file), 'r') as f:
        points = [line for line in f]  # 每行有三个数字，表示一个点
        point_counts.append(len(points))
    
    # 统计 obj 文件中的边数
    with open(os.path.join(obj_folder, obj_file), 'r') as f:
        edges = [line for line in f if line.startswith('l')]  # 每行以 l 开头，表示一个边
        edge_counts.append(len(edges))

# 绘制点数的直方图
plt.figure(figsize=(10, 6))
plt.hist(point_counts, bins=20, color='b', alpha=0.7)
plt.xlabel('Number of Points', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Point Counts per Sample', fontsize=14)
point_hist_path = "point_count_histogram.png"
plt.tight_layout()
plt.savefig(point_hist_path)
plt.show()

# 绘制边数的直方图
plt.figure(figsize=(10, 6))
plt.hist(edge_counts, bins=20, color='g', alpha=0.7)
plt.xlabel('Number of Edges', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Edge Counts per Sample', fontsize=14)
edge_hist_path = "edge_count_histogram.png"
plt.tight_layout()
plt.savefig(edge_hist_path)
plt.show()

# 输出保存路径
print(f"点数直方图已保存为 {point_hist_path}")
print(f"边数直方图已保存为 {edge_hist_path}")
