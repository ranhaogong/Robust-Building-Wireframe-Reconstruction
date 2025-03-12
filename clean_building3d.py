import os
from tqdm import tqdm  # 导入 tqdm 库

# 定义输入和输出文件夹路径
input_folder = '/data/haoran/dataset/building3d/roof/Tallinn/train/xyz'
output_folder = '/data/haoran/dataset/building3d/roof/Tallinn/train/xyz_clean'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取所有.xyz文件
xyz_files = [f for f in os.listdir(input_folder) if f.endswith('.xyz')]

# 使用 tqdm 遍历文件并显示进度条
for filename in tqdm(xyz_files, desc="处理文件中"):
    input_file_path = os.path.join(input_folder, filename)
    output_file_path = os.path.join(output_folder, filename)
    
    # 打开输入文件并读取数据
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()
    
    # 提取每行的前6列数据
    cleaned_lines = []
    for line in lines:
        columns = line.strip().split()
        cleaned_line = ' '.join(columns[:6]) + '\n'
        cleaned_lines.append(cleaned_line)
    
    # 将处理后的数据写入输出文件
    with open(output_file_path, 'w') as outfile:
        outfile.writelines(cleaned_lines)

print("处理完成！")