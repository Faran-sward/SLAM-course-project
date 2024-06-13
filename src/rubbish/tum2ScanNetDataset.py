import os
import shutil
import numpy as np

# 读取输入的txt文件
input_file = 'associate_with_gt.txt'
new_rgb_folder = 'new_rgb'
new_depth_folder = 'new_depth'
pose_folder = 'pose'

# 创建新文件夹
os.makedirs(new_rgb_folder, exist_ok=True)
os.makedirs(new_depth_folder, exist_ok=True)
os.makedirs(pose_folder, exist_ok=True)

def quaternion_to_matrix(qx, qy, qz, qw):
    """将四元数转换为旋转矩阵"""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

with open(input_file, 'r') as file:
    lines = file.readlines()

for idx, line in enumerate(lines):
    parts = line.strip().split()
    timestamp, tx, ty, tz, qx, qy, qz, qw, rgb_time, rgb_path, depth_time, depth_path = parts

    # 复制图像文件到新的文件夹并重命名
    new_rgb_path = os.path.join(new_rgb_folder, f"{idx}.png")
    new_depth_path = os.path.join(new_depth_folder, f"{idx}.png")
    shutil.copy(rgb_path, new_rgb_path)
    shutil.copy(depth_path, new_depth_path)

    # 生成位姿文件
    tx, ty, tz = float(tx), float(ty), float(tz)
    qx, qy, qz, qw = float(qx), float(qy), float(qz), float(qw)
    R = quaternion_to_matrix(qx, qy, qz, qw)
    t = np.array([tx, ty, tz])

    pose_file_path = os.path.join(pose_folder, f"{idx}.txt")
    with open(pose_file_path, 'w') as pose_file:
        for row in np.hstack((R, t.reshape(-1, 1))):
            pose_file.write(' '.join(map('{:.6f}'.format, row)) + '\n')
        pose_file.write('0.000000 0.000000 0.000000 1.000000\n')
