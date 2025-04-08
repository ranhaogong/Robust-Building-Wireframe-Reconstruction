# import os

# # 定义路径
# source_dir = "/data/haoran/dataset/building3d/Point2Roof_tokyo"
# output_file = "/data/haoran/dataset/building3d/Point2Roof_tokyo/train_all.txt"

# # 获取所有文件夹的绝对路径
# folder_paths = []
# for folder_name in os.listdir(source_dir):
#     folder_path = os.path.join(source_dir, folder_name)
#     # 确保是文件夹而非文件
#     if os.path.isdir(folder_path):
#         folder_paths.append(os.path.abspath(folder_path))

# # 检查文件夹列表是否为空
# if not folder_paths:
#     print(f"警告: {source_dir} 中没有找到任何文件夹")
# else:
#     # 写入文件
#     with open(output_file, 'w') as f:
#         for path in folder_paths:
#             f.write(f"{path}\n")
#     print(f"已成功将 {len(folder_paths)} 个文件夹路径写入 {output_file}")

import os

# 定义路径
source_dir = "/data/haoran/dataset/building3d/roof/Entry-level/test/xyz"
output_file = "/data/haoran/dataset/building3d/Point2Roof_Entry/test_all.txt"

# 获取所有 .xyz 文件的绝对路径
xyz_paths = []
for file_name in os.listdir(source_dir):
    if file_name.endswith('.xyz'):  # 只选择 .xyz 文件
        file_path = os.path.join(source_dir, file_name)
        if os.path.isfile(file_path):  # 确保是文件
            xyz_paths.append(os.path.abspath(file_path))

# 检查文件列表是否为空
if not xyz_paths:
    print(f"警告: {source_dir} 中没有找到任何 .xyz 文件")
else:
    # 写入文件
    with open(output_file, 'w') as f:
        for path in xyz_paths:
            f.write(f"{path}\n")
    print(f"已成功将 {len(xyz_paths)} 个 .xyz 文件路径写入 {output_file}")