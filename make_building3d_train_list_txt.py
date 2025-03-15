import os

def write_folder_paths(input_dir, output_file):
    """
    读取指定目录下的所有文件夹，并将绝对路径写入指定文件。
    
    参数：
    input_dir (str): 输入目录路径
    output_file (str): 输出文件路径
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在！")
        return
    
    # 获取输入目录下的所有子目录（文件夹）
    folder_list = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    # 如果没有子目录，打印提示并退出
    if not folder_list:
        print(f"警告：目录 {input_dir} 下没有子文件夹！")
        return
    
    # 对文件夹列表排序（可选，确保输出顺序一致）
    folder_list.sort()
    
    # 将绝对路径写入文件
    with open(output_file, 'w') as f:
        for folder in folder_list:
            folder_abs_path = os.path.abspath(os.path.join(input_dir, folder))
            f.write(f"{folder_abs_path}\n")
    
    print(f"成功将 {len(folder_list)} 个文件夹路径写入 {output_file}")

def main():
    # 设置输入和输出路径
    input_dir = "/data/haoran/dataset/building3d/Point2Roof_tokyo_seg"
    output_file = "/data/haoran/dataset/building3d/Point2Roof_tokyo_seg/train_all.txt"
    
    # 执行写入操作
    write_folder_paths(input_dir, output_file)

if __name__ == "__main__":
    main()