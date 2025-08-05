import os
import pandas as pd
import shutil

# 创建目标文件夹
target_folder = 'landpointCollection'
os.makedirs(target_folder, exist_ok=True)

# 遍历当前文件夹及其子文件夹
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('_land_point.csv'):
            # 获取文件的父文件夹名称
            parent_folder_name = os.path.basename(root)
            # 构建源文件路径
            source_file = os.path.join(root, file)
            # 创建新文件名
            new_file_name = f"{parent_folder_name}_{file}"
            # 复制文件到目标文件夹并重命名
            shutil.copy(source_file, os.path.join(target_folder, new_file_name))
            # 读取 CSV 文件并修改列名
            df = pd.read_csv(os.path.join(target_folder, new_file_name))
            df.columns.values[0] = 'X'  # 修改第一列列名
            df.columns.values[1] = 'Y'  # 修改第二列列名
            # 保存修改后的 CSV 文件
            df.to_csv(os.path.join(target_folder, new_file_name), index=False)

print("文件复制和列名修改完成。")