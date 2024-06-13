import cv2
import numpy as np
import os
import csv


def read_filenames_from_folder(folder_path):
    # 定义一个自定义的排序函数，按数字大小比较
    def numerical_sort(file):
        return int(os.path.basename(file).split('.')[0])

    # 获取文件夹中所有文件的路径，并按数字大小排序
    filenames = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, file))]
    filenames.sort(key=numerical_sort)
    return filenames


# 读取深度图像
def read_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Error: Unable to load depth image at {path}")
    return depth


# 特征匹配函数
def find_feature_matches(method, img1, img2):
    if method == "ORB":
        feature_detector = cv2.ORB_create()
        norm_type = cv2.NORM_HAMMING
    elif method == "SIFT":
        feature_detector = cv2.xfeatures2d.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method == "SURF":
        feature_detector = cv2.xfeatures2d.SURF_create()
        norm_type = cv2.NORM_L2
    else:
        raise ValueError("Unsupported feature detection method")

    keypoints1, descriptors1 = feature_detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = feature_detector.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 筛选匹配点
    min_dist = min(matches, key=lambda x: x.distance).distance
    max_dist = max(matches, key=lambda x: x.distance).distance
    print(f"min_dist: {min_dist}, max_dist: {max_dist}")

    if method == "ORB":
        good_matches = [m for m in matches if m.distance <= max(2 * min_dist, 50.0)]
    elif method == "SURF":
        good_matches = [m for m in matches if m.distance <= max(2 * min_dist, 0.15)]
    else:  # sift
        good_matches = [m for m in matches if m.distance <= max(2 * min_dist, 100)]

    print(len(good_matches), len(matches))
    return keypoints1, keypoints2, good_matches, min_dist, max_dist


# 位姿估计函数
def pose_estimation_2d2d(keypoints1, keypoints2, matches, K):
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    essential_matrix, _ = cv2.findEssentialMat(points1, points2, K)
    _, R, t, _ = cv2.recoverPose(essential_matrix, points1, points2, K)

    return R, t


# 像素坐标转换到相机坐标
def pixel2cam(point, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (point[0] - cx) / fx
    y = (point[1] - cy) / fy
    return np.array([x, y])


# 三角化函数
def triangulate_points(R, t, K, keypoints1, keypoints2, matches):
    pts_1 = np.float32([pixel2cam(keypoints1[m.queryIdx].pt, K) for m in matches])
    pts_2 = np.float32([pixel2cam(keypoints2[m.trainIdx].pt, K) for m in matches])

    T1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    T2 = np.hstack((R, t))

    pts_4d = cv2.triangulatePoints(T1, T2, pts_1.T, pts_2.T)
    pts_4d /= pts_4d[3]

    points = [pts_4d[:3, i] for i in range(pts_4d.shape[1])]
    return points


# 计算误差函数
def compute_depth_error(triangulated_points, depth_img, keypoints1, matches, K):
    errors = []
    fx, _, cx, _, fy, cy, _, _, _ = K.flatten()

    for i, match in enumerate(matches):
        u, v = keypoints1[match.queryIdx].pt
        u, v = int(u), int(v)
        if u < 0 or v < 0 or u >= depth_img.shape[1] or v >= depth_img.shape[0]:
            continue
        depth = depth_img[v, u] / 5000.0  # Assuming depth is in millimeters and scale factor is 5000
        if depth <= 0:
            continue

        X, Y, Z = triangulated_points[i]
        Z_est = Z

        error = abs(depth - Z_est)
        errors.append(error)

    errors = np.array(errors)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    rmse_log = np.sqrt(np.mean(np.log(1 + error) ** 2))

    return mae, rmse, rmse_log


# 主函数
def main():
    folder_path = "Portland_hotel/output_image"
    depth_folder_path = "Portland_hotel/output_depth"
    base_output_csv = "depth_errors.csv"
    interval = 120
    selected_pictures = 200

    filenames = read_filenames_from_folder(folder_path)
    depth_filenames = read_filenames_from_folder(depth_folder_path)

    K = np.array([[570.342224, 0, 320],
                  [0, 570.342224, 240],
                  [0, 0, 1]])

    results = []
    (total_mae, total_rmse, total_rmse_log) = (0, 0, 0)
    for item in ["ORB", "SURF", 'SIFT']:
        # for item in ["ORB"]:
        output_csv = item + "_" + base_output_csv

        for i in range(0, min(selected_pictures, len(filenames) - interval)):
            img1 = cv2.imread(filenames[i])
            img2 = cv2.imread(filenames[i + interval])
            # print("rgb img loaded.")

            depth_img1 = read_depth(depth_filenames[i])
            depth_img2 = read_depth(depth_filenames[i + interval])
            # print("depth img loaded.")

            keypoints1, keypoints2, matches, min_dist, max_dist = find_feature_matches(item, img1, img2)
            # keypoints1, keypoints2, matches = find_surf_feature_matches(img1, img2)
            # print("feature got.")
            if len(matches) <= 5:
                continue
            R, t = pose_estimation_2d2d(keypoints1, keypoints2, matches, K)

            triangulated_points = triangulate_points(R, t, K, keypoints1, keypoints2, matches)

            print(f"triangulation op finished, num of points: {len(triangulated_points)}")

            mae, rmse, rmse_log = compute_depth_error(triangulated_points, depth_img2, keypoints1, matches, K)
            print(f"mae: {mae}, rmse: {rmse}, rmse_log: {rmse_log}")
            total_mae += mae
            total_rmse += rmse
            total_rmse_log += rmse_log

            # print("error computed.")

            results.append([filenames[i], filenames[i + interval], mae, rmse, rmse_log])

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Image1", "Image2", "MAE", "RMSE", "RMSE log"])
            writer.writerows(results)

            avg_mae = total_mae / len(results)
            avg_rmse = total_rmse / len(results)
            avg_rmse_log = total_rmse_log / len(results)
            writer.writerow(["Average", "", avg_mae, avg_rmse, avg_rmse_log])

        print(f"Results saved to {output_csv}")
        print(f"Average MAE: {avg_mae}, Average RMSE: {avg_rmse}, Average RMSE log: {avg_rmse_log}")


if __name__ == "__main__":
    main()
