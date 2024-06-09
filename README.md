# SLAM-course-project
 同济大学软件学院SLAM期末项目

## Environment

### C++ environment construction

The c++ code can only work in opencv with the `vision>=4.2`, version 4.2.0 is encouraged because of the support of `SIFT` and `SURF` detect method. You can construct the opencv referring to the command below:

```bash
# download OpenCV source code
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.9.0

# download opencv_contrib source code
cd ..
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 4.9.0

# build opencv with contribute mode
cd ../opencv
mkdir build
cd build

# cmake constructino
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DOPENCV_ENABLE_NONFREE=ON ..
make -j$(nproc)
sudo make install

# check if opencv has been installed successfully
pkg-config --modversion opencv4

# if you encountered problem in cmake build operation of ninja
sudo apt install ninja-build
```

> references:
>
> https://blog.csdn.net/Gordon_Wei/article/details/88920411
>
> https://blog.csdn.net/AiXiangSiyou/article/details/121629190

### Python environment construction

The python code also need the avaliability of `SIFT` and `SURF` detect method, so `opencv-python` is needed with the `version<=3.4`, you can just use the command

```bash
# python 3.8
pip install opencv-python==3.4.8.29
pip install opencv-contrib-python==3.4.8.29
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib Path
```

or using conda environment export file `deltas.yaml`:

```bash
conda env create -f deltas.yaml
```

## Dataset

this project used `Portland_hotel` and `tum` dataset:

### Download links:

> tum fr1/xyz dataset:
>
> https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
>
> Portland_hotel dataset:
>
> https://sun3d.cs.princeton.edu/data/Portland_hotel/

### Dataset format:

tum dataset:

```
rgbd_dataset_freiburg1_xyz
├── accelerometer.txt
├── associate.txt  # offcial tools generated
├── depth
│   ├── 1305031102.160407.png
│   ├── ...
│   └── 1305031128.754646.png
├── depth.txt
├── groundtruth.txt
├── rgb
│   ├── 1305031102.175304.png
│   ├── ...
│   └── 1305031128.747363.png
└── rgb.txt
```

Portland_hotel dataset:

```
Portland_hotel
├── depth
│   ├── 0000001-000000000000.png
│   ├── ...
│   └── 0013323-000446500440.png
├── extrinsics
│   └── 20140808220511.txt
├── image
│   ├── 0000001-000000000000.jpg
│   ├── ...
│   └── 0013323-000446500152.jpg
├── intrinsics.txt
└── thumbnail
    └── 20140808220511.jpg
```

## Run the code

### Triangulation part

c++ version: 

compile cmake project to execute `triangulation` binary file.

python version: 

```bash
python main.py
```

and you can get 3 `.csv` files to look through the result.

### Deltas part

> https://github.com/magicleap/DELTAS
>
> https://blog.csdn.net/qq_29462849/article/details/118586745

```bash
python test_learnabledepth.py
```

## Result

|          |  Avg MAE   |  Avg RMSE   | Avg RMSE log |
| :------: | :--------: | :---------: | :----------: |
|   ORB    |   58.997   |   364.173   |    2.092     |
|   SIFT   |   48.187   |   244.912   |    2.171     |
| **SURF** | **41.150** | **222.978** |  **2.073**   |

Table 1. Triangulation using 3 methods in `200` image-pairs with the interval of `120` frames.

|          |  AbR  |  SqR  | AbD/MAE | RMSE  | RMSE log |
| :------: | :---: | :---: | :-----: | :---: | :------: |
| 1 frame  | 0.802 | 3.408 |  4.209  | 4.349 |  1.647   |
| 2 frames | 0.802 | 3.390 |  4.199  | 4.325 |  1.643   |
| 4 frames | 0.803 | 3.378 |  4.192  | 4.312 |  1.639   |
| 5 frames | 0.803 | 3.381 |  4.194  | 4.313 |  1.642   |
| 7 frames | 0.803 | 3.379 |  4.193  | 4.311 |  1.640   |

Table 2. Performance of depth estimation (pretrained model) in tum dataset using sequences of length `3`.

## Reference

> [Inference Code for DELTAS: Depth Estimation by Learning Triangulation And densification of Sparse point (ECCV 2020)s](https://github.com/magicleap/DELTAS)
>
> [DELTAS: Depth Estimation by Learning Triangulation And densification of Sparse points](https://arxiv.org/abs/2003.08933)