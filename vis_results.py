import os
import numpy as np
from tqdm import tqdm

def process_files(save_dir, xyz_dir, output_dir):
    # 创建输出目录如果不存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有 .obj 文件
    obj_files = [f for f in os.listdir(save_dir) if f.endswith('.obj')]
    
    # 使用 tqdm 显示进度条
    for obj_file in tqdm(obj_files, desc="Processing files"):
        # 读取 obj 文件中的顶点
        obj_vertices = []
        obj_path = os.path.join(save_dir, obj_file)
        
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    coords = [float(x) for x in line.strip().split()[1:4]]
                    obj_vertices.append(coords)
        
        obj_vertices = np.array(obj_vertices)
        
        # 读取对应的 xyz 文件
        xyz_file = os.path.join(xyz_dir, obj_file.replace('.obj', '.xyz'))
        if not os.path.exists(xyz_file):
            print(f"Warning: Corresponding .xyz file not found for {obj_file}")
            continue
        
        xyz_points = np.loadtxt(xyz_file)[:, :3]  # 只取前三列(xyz)
        
        # 创建输出文件（改为 .xyz 格式）
        output_file = os.path.join(output_dir, obj_file.replace('.obj', '.xyz'))
        
        with open(output_file, 'w') as f:
            # 写入 obj 的顶点(红色)
            for v in obj_vertices:
                # x y z r g b (红色: 255 0 0)
                f.write(f"{v[0]} {v[1]} {v[2]} 255 0 0\n")
            
            # 写入 xyz 的点(黑色)
            for p in xyz_points:
                # x y z r g b (黑色: 0 0 0)
                f.write(f"{p[0]} {p[1]} {p[2]} 255 255 255\n")

def main():
    # 设置路径
    save_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_wo_fintune_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_lovasz_edge_loss/save"
    xyz_dir = "/data/haoran/dataset/building3d/tokyo/testing/xyz"
    output_dir = "/data/haoran/Point2Roof/output/building3d_tokyo_wo_fintune_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_lovasz_edge_loss/save_vis"
    
    process_files(save_dir, xyz_dir, output_dir)

if __name__ == "__main__":
    main()