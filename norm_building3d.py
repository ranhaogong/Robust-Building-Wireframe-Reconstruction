import os
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库

# 定义输入和输出文件夹路径
input_folder = '/data/haoran/Point2Roof/output/building3d_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_lovasz_wavelet_edge/vis'
output_folder = '/data/haoran/Point2Roof/output/building3d_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_lovasz_wavelet_edge/vis_norm'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取所有.xyz文件
xyz_files = [f for f in os.listdir(input_folder) if f.endswith('.xyz')]

# 使用 tqdm 遍历文件并显示进度条
for filename in tqdm(xyz_files, desc="归一化文件中"):
    input_file_path = os.path.join(input_folder, filename)
    output_file_path = os.path.join(output_folder, filename)
    
    # 读取文件数据
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()
    
    # 提取前三列数据 (x, y, z)
    xyz_data = []
    color_data = []
    for line in lines:
        columns = line.strip().split()
        xyz_data.append([float(columns[0]), float(columns[1]), float(columns[2])])
        color_data.append([float(columns[3]), float(columns[4]), float(columns[5])])
    
    # 将数据转换为 NumPy 数组
    xyz_data = np.array(xyz_data)
    color_data = np.array(color_data)
    
    min_pt, max_pt = np.min(xyz_data, axis=0), np.max(xyz_data, axis=0)
    maxXYZ = np.max(max_pt)
    minXYZ = np.min(min_pt)
    min_pt[:] = minXYZ
    max_pt[:] = maxXYZ
    centroid = np.mean(xyz_data, axis=0)
    xyz_data -= centroid
    max_distance = np.max(np.linalg.norm(xyz_data, axis=1))
    xyz_data /= max_distance
    
    # color_data = color_data / 255.0  # 将 RGB 值从 [0, 255] 归一化到 [0, 1]
    # color_data = (color_data - color_data.mean(axis=0)) / color_data.std(axis=0)  # 标准化（均值为 0，标准差为 1）
    
    # 将归一化后的数据写回文件
    with open(output_file_path, 'w') as outfile:
        for (x, y, z), (r, g, b) in zip(xyz_data, color_data):
            # 保留6位小数
            outfile.write(f"{x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

print("归一化处理完成！")