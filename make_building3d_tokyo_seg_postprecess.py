import os

def check_obj_file(file_path):
    """
    检查 .obj 文件中以 'v' 开头的行数和以 'l' 开头的行数。
    如果任意一个为 0，则返回 True 表示有问题。
    
    参数：
    file_path (str): .obj 文件的路径
    
    返回：
    bool: 如果文件有问题（v_count 或 l_count 为 0），返回 True，否则返回 False
    """
    v_count = 0  # 统计顶点数
    l_count = 0  # 统计边数
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):  # 注意 'v ' 而不是 'v'，以排除 'vn' 等
                    v_count += 1
                elif line.startswith('l '):  # 注意 'l '，以确保只统计边
                    l_count += 1
        
        # 检查是否为 0
        if v_count == 0 or l_count == 0:
            print(f"文件: {file_path}")
            print(f"  顶点数 (v): {v_count}")
            print(f"  边数 (l): {l_count}")
            print("警告：顶点数或边数为 0！")
            return True
        return False
    
    except Exception as e:
        print(f"错误：无法读取文件 {file_path}，原因：{e}")
        return True  # 将读取错误的文件也视为有问题

def delete_files(obj_file_path, xyz_dir):
    """
    删除指定的 .obj 文件，并删除 xyz_dir 中对应的同名 .xyz 文件。
    
    参数：
    obj_file_path (str): .obj 文件的路径
    xyz_dir (str): .xyz 文件所在的目录
    """
    try:
        # 删除 .obj 文件
        os.remove(obj_file_path)
        print(f"已删除 .obj 文件: {obj_file_path}")
    except Exception as e:
        print(f"错误：无法删除 .obj 文件 {obj_file_path}，原因：{e}")
    
    # 构造对应的 .xyz 文件路径
    file_name = os.path.basename(obj_file_path)  # 获取文件名（包含扩展名）
    xyz_file_name = os.path.splitext(file_name)[0] + ".xyz"  # 替换扩展名为 .xyz
    xyz_file_path = os.path.join(xyz_dir, xyz_file_name)
    
    try:
        if os.path.exists(xyz_file_path):
            os.remove(xyz_file_path)
            print(f"已删除对应的 .xyz 文件: {xyz_file_path}")
        else:
            print(f"警告：对应的 .xyz 文件 {xyz_file_path} 不存在，跳过删除")
    except Exception as e:
        print(f"错误：无法删除 .xyz 文件 {xyz_file_path}，原因：{e}")

def traverse_obj_files(obj_dir, xyz_dir):
    """
    遍历指定目录中的所有 .obj 文件，并检查每个文件的顶点数和边数。
    如果文件有问题，删除该文件并删除对应的 .xyz 文件。
    统计有问题的文件总数。
    
    参数：
    obj_dir (str): .obj 文件所在的目录路径
    xyz_dir (str): .xyz 文件所在的目录路径
    """
    # 确保目录存在
    if not os.path.exists(obj_dir):
        print(f"错误：.obj 文件目录 {obj_dir} 不存在！")
        return
    if not os.path.exists(xyz_dir):
        print(f"错误：.xyz 文件目录 {xyz_dir} 不存在！")
        return
    
    # 初始化计数器
    total_files = 0  # 总文件数
    problem_files = 0  # 有问题的文件数
    
    # 遍历目录中的所有文件
    obj_files_found = False
    for root, _, files in os.walk(obj_dir):
        for file in files:
            if file.endswith('.obj'):
                obj_files_found = True
                total_files += 1
                file_path = os.path.join(root, file)
                if check_obj_file(file_path):
                    problem_files += 1
                    delete_files(file_path, xyz_dir)
    
    if not obj_files_found:
        print(f"警告：目录 {obj_dir} 下没有找到任何 .obj 文件！")
    else:
        print(f"检查完成！")
        print(f"总共检查了 {total_files} 个 .obj 文件")
        print(f"其中有 {problem_files} 个文件存在问题（顶点数或边数为 0，或无法读取），已删除相关文件")

def main():
    # 设置要遍历的目录
    obj_dir = "/data/haoran/dataset/building3d/tokyo/training_seg/wireframe"
    xyz_dir = "/data/haoran/dataset/building3d/tokyo/training_seg/xyz"
    
    # 执行遍历和检查
    print(f"开始检查目录 {obj_dir} 中的 .obj 文件，并删除有问题的文件及其对应的 .xyz 文件...")
    traverse_obj_files(obj_dir, xyz_dir)

if __name__ == "__main__":
    main()