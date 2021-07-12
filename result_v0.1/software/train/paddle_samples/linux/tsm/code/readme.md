<!-- omit in toc -->
# paddle TSM 性能复现


此处给出了[paddle TSM](https://github.com/PaddlePaddle/PaddleVideo)性能和精度测试任务的详细流程，包括执行环境、paddle版本、环境搭建、测试脚本、测试结果和测试日志。

<!-- omit in toc -->
## 目录
- [一、环境介绍](#一环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [二、环境搭建](#二环境搭建)
  - [1. 单机（单卡、8卡）环境搭建](#1-单机单卡8卡环境搭建)
  - [2. 多机（32卡）环境搭建](#2-多机32卡环境搭建)
  - [3. 数据说明](#3-数据说明)

- [三、测试步骤](#三测试步骤)
- [四、测试结果](#四测试结果)
- [五、日志数据](#五日志数据)
  - [1. 4机32卡日志](#1日志)


## 一、环境介绍

### 1.物理机环境

我们使用了同一个物理机环境，对 [TSM paddle](https://github.com/PaddlePaddle/PaddleVideo) 模型进行了测试,具体参数如下：
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

本次测试所以用的docker镜像相关信息如下所示：

- **镜像版本**: `registry.baidu.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.0-cudnn8`
- **paddle 版本**: `develop`
- **CUDA 版本**: `11.0`
- **cuDnn 版本**: `8.0.4`

## 二、环境搭建

### 1. 单机（单卡、8卡）环境搭建

单机环境搭建主要过程如下:
- 新建docker container:
使用`registry.baidu.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.0-cudnn8`docker镜像创建docker容器  

- 参考[TSM paddle实现](https://github.com/PaddlePaddle/PaddleVideo) 安装依赖
    ```bash
    tqdm
    PyYAML>=5.1
    numpy
    decord
    pandas
    opencv-python==4.2.0.32
    paddlepaddle[develop]
    ```

- **拉取代码**

    ```bash
    git clone https://github.com/PaddlePaddle/PaddleVideo.git
    ```

### 2. 多机（32卡）环境搭建

多机环境同单机环境一致


### 3. 数据说明
- 具体数据使用情况，请参看[DATA.md](DATA.md)

## 三、测试步骤

- **单卡启动脚本：**

    若测试单机1卡AMP的训练性能和精度，执行如下命令：

    ```bash
       export FLAGS_conv_workspace_size_limit=1000 #MB
       export FLAGS_cudnn_exhaustive_search=1
       export FLAGS_cudnn_batchnorm_spatial_persistent=1
       export CUDA_VISIBLE_DEVICES=0
       python3 main.py  -c configs/recognition/tsm/tsm_k400_frames.yaml --amp --validate
    ```

- **8卡启动脚本：**

    若测试单机8卡AMP的训练性能和精度，执行如下命令：

    ```bash
       export FLAGS_conv_workspace_size_limit=1000 #MB
       export FLAGS_cudnn_exhaustive_search=1
       export FLAGS_cudnn_batchnorm_spatial_persistent=1
       python3 -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 --log_dir=log_tsm main.py --amp --validate -c configs/recognition/tsm/tsm_k400_frames.yaml
    ```

- **32卡启动脚本**
    32卡启动脚本如8卡启动脚本，即分别在4个节点上分别运行8卡的启动脚本。

## 四、测试结果
(由于单卡运行时间周期过长，此处只给出单卡训练的部分日志数据)
- AMP结果

|  训练卡数   | Time2train(sec)  |吞吐(samples/sec)  |  准确率(%) |
|------------|------------|------------|------------|
|    1卡     |     -      |    80.222     |    -     |
|    8卡     | 22549.33   |   549.176      |     71.00     |
|    32卡    | 5909.33   |   2042.24      |    71.02     |
  

## 五、日志数据
### 1.日志
- [4机32卡、AMP ](../log/GPUx32_AMP.log)
- [1机8卡、AMP ](../log/GPUx8_AMP.log)
- [1机1卡、AMP ](../log/GPUx1_AMP.log)

