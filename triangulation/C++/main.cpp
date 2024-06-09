#include <iostream>
#include <vector>
#include <string>
#include <filesystem> // C++17 标准库
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
// #include <opencv2/xfeatures2d/nonfree.hpp> //OpenCV 4.2.0 及之后版本

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void find_ORB_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);
  
void find_SIFT_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

void find_SURF_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

void readFilenamesFromFolder(const std::string& folderPath, std::vector<std::string>& filenames) {
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            filenames.push_back(entry.path().string());
        }
    }
}

Mat readDepth(const std::string &path) {
    Mat depth = imread(path, IMREAD_UNCHANGED);
    if (depth.empty()) {
        cerr << "Error: Unable to load depth image at " << path << endl;
        exit(EXIT_FAILURE);
    }
    return depth;
}

void pose_estimation_2d2d(
  const std::vector<KeyPoint> &keypoints_1,
  const std::vector<KeyPoint> &keypoints_2,
  const std::vector<DMatch> &matches,
  Mat &R, Mat &t) {
  // 相机内参,TUM Freiburg2
  Mat K = (Mat_<double>(3, 3) << 570.342224, 0, 320, 0, 570.342224, 240, 0, 0, 1);

  //-- 把匹配点转换为vector<Point2f>的形式
  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

//-- 计算本质矩阵
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, K);
    std::cout << "Essential Matrix: \n" << essential_matrix << std::endl;

//-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose(essential_matrix, points1, points2, K, R, t);

    std::cout << "Rotation Matrix: \n" << R << std::endl;
    std::cout << "Translation Vector: \n" << t << std::endl;
}

Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points) {
  Mat T1 = (Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
  Mat T2 = (Mat_<float>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
  );

  Mat K = (Mat_<double>(3, 3) << 570.342224, 0, 320, 0, 570.342224, 240, 0, 0, 1);
  vector<Point2f> pts_1, pts_2;
  for (DMatch m:matches) {
    // 将像素坐标转换至相机坐标
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
  }

  Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

  // 转换成非齐次坐标
  for (int i = 0; i < pts_4d.cols; i++) {
    Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0); // 归一化
    Point3d p(
      x.at<float>(0, 0),
      x.at<float>(1, 0),
      x.at<float>(2, 0)
    );
    points.push_back(p);
  }
}

Mat calculateDepthError(const vector<Point3d> &points4D, const Mat &depthMap, const Mat &K) {
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    
    Mat depthErrors(1, points4D.size(), CV_64F);
    // cout << 1 << endl;
    for (int i = 0; i < points4D.size(); ++i) {
        double X = points4D[i].x;
        double Y = points4D[i].y;
        double Z = points4D[i].z;
        
        int u = static_cast<int>(fx * X / Z + cx);
        int v = static_cast<int>(fy * Y / Z + cy);

        if (u >= 0 && u < depthMap.cols && v >= 0 && v < depthMap.rows) {
            double depth = static_cast<double>(depthMap.at<unsigned short>(u, v)) / 5000.0;
            if (depth > 0) {
                depthErrors.at<double>(i) = abs(Z - depth);
            } else {
                depthErrors.at<double>(i) = NAN;
                cout << "depth not measured." << endl;
            }
        } else {
            depthErrors.at<double>(i) = NAN;
            cout << "pixel error: (u, v) = " << u << ", " << v << endl;
        }
    }
    return depthErrors;
}

void computeErrors(const Mat &depthErrors) {
    double mae = 0;
    double rmse = 0;
    double rmse_log = 0;
    int count = 0;

    for (int i = 0; i < depthErrors.cols; ++i) {
        double error = depthErrors.at<double>(i);
        if (!isnan(error)) {
            mae += error;
            rmse += error * error;
            rmse_log += log(1 + error) * log(1 + error);
            ++count;
        }
    }
    if (!count) {
      cout << "compute error op failed." << endl;
      return;
      }

    mae /= count;
    rmse = sqrt(rmse / count);
    rmse_log = sqrt(rmse_log / count);

    cout << "MAE: " << mae << endl;
    cout << "RMSE: " << rmse << endl;
    cout << "RMSE log: " << rmse_log << endl;
}

void processImagesAndDepths(const vector<string> &imagePaths, const vector<string> &depthPaths, const Mat &K, int interval=20) {
    for (size_t i = 0; i < imagePaths.size() - interval; ++i) {

        Mat img1 = imread(imagePaths[i]);
        Mat img2 = imread(imagePaths[i + interval]);
        cout << "rgb img loaded." << endl;

        Mat depth1 = readDepth(depthPaths[i]);
        Mat depth2 = readDepth(depthPaths[i + interval]);
        cout << " depth img loaded." << endl;

        vector<KeyPoint> keypoints1, keypoints2;
        vector<DMatch> matches;
        Mat E, mask, R, t;

        find_ORB_feature_matches(img1, img2, keypoints1, keypoints2, matches);
        cout << "feature got." << endl;

        pose_estimation_2d2d(keypoints1, keypoints2, matches, R, t);
        vector<Point3d> points;
        triangulation(keypoints1, keypoints2, matches, R, t, points);

        cout << "triangulation op finished, num of points: " << points.size() << endl;

        Mat depthErrors = calculateDepthError(points, depth2, K);
        cout << "error computed." << endl;
        computeErrors(depthErrors);
    }
}

int main() {
    string imageFolderPath = "Portland_hotel/output_image";
    string depthFolderPath = "Portland_hotel/output_depth";

    vector<string> imagePaths;
    vector<string> depthPaths;

    readFilenamesFromFolder(imageFolderPath, imagePaths);
    readFilenamesFromFolder(depthFolderPath, depthPaths);

    cout << "image path loaded." << endl;
    cout << "rgb: " << imagePaths.size() << endl;
    cout << "depth: " << depthPaths.size() << endl;

    Mat K = (Mat_<double>(3, 3) << 570.342224, 0, 320, 0, 570.342224, 240, 0, 0, 1);

    processImagesAndDepths(imagePaths, depthPaths, K);

    return 0;
}

void find_ORB_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();

  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

void find_SIFT_feature_matches(const Mat &img_1, const Mat &img_2,
                               std::vector<KeyPoint> &keypoints_1,
                               std::vector<KeyPoint> &keypoints_2,
                               std::vector<DMatch> &matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    Ptr<SIFT> sift = SIFT::create();
    
    //-- 第一步:检测 SIFT 关键点位置
    sift->detect(img_1, keypoints_1);
    sift->detect(img_2, keypoints_2);

    //-- 第二步:根据关键点位置计算 SIFT 描述子
    sift->compute(img_1, keypoints_1, descriptors_1);
    sift->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的SIFT描述子进行匹配，使用 L2 距离
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    // 找出所有匹配之间的最小距离和最大距离，即最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。但有时候最小距离会非常小，设置一个经验值30作为下限
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

void find_SURF_feature_matches(const Mat &img_1, const Mat &img_2,
                               std::vector<KeyPoint> &keypoints_1,
                               std::vector<KeyPoint> &keypoints_2,
                               std::vector<DMatch> &matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    Ptr<SURF> surf = SURF::create();

    //-- 第一步:检测 SURF 关键点位置
    surf->detect(img_1, keypoints_1);
    surf->detect(img_2, keypoints_2);

    //-- 第二步:根据关键点位置计算 SURF 描述子
    surf->compute(img_1, keypoints_1, descriptors_1);
    surf->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的SURF描述子进行匹配，使用 L2 距离
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    // 找出所有匹配之间的最小距离和最大距离，即最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。但有时候最小距离会非常小，设置一个经验值30作为下限
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

