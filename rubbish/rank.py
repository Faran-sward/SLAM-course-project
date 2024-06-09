import os
import shutil

def get_timestamp(filename):
    # 从文件名中提取时间戳，例如： "image_1234567890.jpg" -> 1234567890
    return int(filename.split('-')[0])

def assign_depth_list(image_list, depth_list):
    depth_temp = depth_list.copy()
    depth_list.clear()

    idx = 0
    depth_time = get_timestamp(depth_temp[idx])
    time_low = depth_time

    for image_file in image_list:
        image_time = get_timestamp(image_file)

        while depth_time < image_time:
            if idx == len(depth_temp) - 1:
                break

            time_low = depth_time
            depth_time = get_timestamp(depth_temp[idx + 1])
            idx += 1

        if idx == 0 and depth_time > image_time:
            depth_list.append(depth_temp[idx])
            continue

        if abs(image_time - time_low) < abs(depth_time - image_time):
            depth_list.append(depth_temp[idx - 1])
        else:
            depth_list.append(depth_temp[idx])

def rename_and_copy_files(image_dir, depth_dir, output_image_dir, output_depth_dir):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_depth_dir):
        os.makedirs(output_depth_dir)

    image_files = sorted(os.listdir(image_dir))
    depth_files = sorted(os.listdir(depth_dir))

    assign_depth_list(image_files, depth_files)

    for i, (image_file, depth_file) in enumerate(zip(image_files, depth_files)):
        new_image_name = f"{i}.jpg"
        new_depth_name = f"{i}.png"

        shutil.copy(os.path.join(image_dir, image_file), os.path.join(output_image_dir, new_image_name))
        shutil.copy(os.path.join(depth_dir, depth_file), os.path.join(output_depth_dir, new_depth_name))

# 示例路径，可以根据实际情况进行调整
image_dir = "Portland_hotel/image"
depth_dir = "Portland_hotel/depth"
output_image_dir = "Portland_hotel/output_image"
output_depth_dir = "Portland_hotel/output_depth"

rename_and_copy_files(image_dir, depth_dir, output_image_dir, output_depth_dir)
