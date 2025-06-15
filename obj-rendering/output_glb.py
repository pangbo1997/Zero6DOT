import os
import json

def find_glb_files(root_dir):
    glb_files = []

    # 遍历root_dir及其所有子目录
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            # 如果文件以.glb结尾，将其完整路径添加到列表中
            if file_name.endswith('.glb'):
                full_path = os.path.join(root, file_name)
                glb_files.append(full_path)

    return glb_files

root_dir = "/data/hf-objaverse-v1/glbs"
glb_files = find_glb_files(root_dir)

# 将列表保存为JSON文件
output_file = "glb_paths.json"
with open(output_file, 'w') as file:
    json.dump(glb_files, file)

print(f"{len(glb_files)} .glb files found and saved to {output_file}")
