import os
import shutil
from tqdm import tqdm

# 定义路径
data_dir = "/data/haoran/dataset/building3d/tokyo/training"
obj_dir = os.path.join(data_dir, "wireframe")
xyz_dir = os.path.join(data_dir, "xyz")
output_dir = "/data/haoran/dataset/building3d/Point2Roof_tokyo"

# 创建目标文件夹
os.makedirs(output_dir, exist_ok=True)

# 获取所有 obj 文件
obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]

# 使用 tqdm 显示进度条
for obj_file in tqdm(obj_files, desc="Processing files"):
    # 提取建筑单体名称 xxx
    building_name = os.path.splitext(obj_file)[0]

    # 构建对应的 xyz 文件路径
    xyz_file = f"{building_name}.xyz"
    xyz_file_path = os.path.join(xyz_dir, xyz_file)

    # 检查 xyz 文件是否存在
    if not os.path.exists(xyz_file_path):
        print(f"Warning: Corresponding .xyz file for {obj_file} not found.")
        continue

    # 创建建筑单体文件夹
    building_dir = os.path.join(output_dir, building_name)
    os.makedirs(building_dir, exist_ok=True)

    # 定义目标文件路径
    target_obj = os.path.join(building_dir, "polygon.obj")
    target_xyz = os.path.join(building_dir, "points.xyz")

    # 复制并重命名文件
    shutil.copy(os.path.join(obj_dir, obj_file), target_obj)
    shutil.copy(xyz_file_path, target_xyz)

print("All files processed.")
