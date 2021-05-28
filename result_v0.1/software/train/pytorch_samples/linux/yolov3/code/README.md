# Gluon-CV YOLOv3 性能测试

此处给出了基于 [Gluon-CV](https://github.com/dmlc/gluon-cv) 实现的 YOLOv3 任务的详细复现流程，包括环境介绍、环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [Gluon CV YOLOv3 性能测试](#Gluon-CV-YOLOv3-性能测试)
  - [一、环境搭建](#一环境搭建)
      - [1.单机（单卡、8卡）环境搭建](#1单机单卡8卡环境搭建)
  - [二、数据准备](#二数据准备)
  - [三、测试步骤](#三测试步骤)
      - [1.单卡吞吐测试](#1单卡吞吐测试)
      - [2.八卡Time2Train吞吐测试](#2八卡Time2Train吞吐测试)
  - [四、日志数据](#四日志数据)
  - [五、性能指标](#五性能指标)


## 一、环境搭建

### 1.单机（单卡、8卡）环境搭建

单机环境的搭建，我们遵循了 GluonCV 官网提供的 [Install](https://cv.gluon.ai/install.html)教程成功搭建了测试环境，主要过程如下：


- 下载Gluon CV repo,并进入目录

   ```bash
   git clone https://github.com/dmlc/gluon-cv.git
   cd gluon-cv
   # 本次测试是在如下版本下完成的：
   git checkout 026d39c16a671ff01dc7437ac9392bfafa26550e
   ```

- 制作Docker镜像

   ```bash
   cd tools/docker/
   
   docker build --build-arg USER_ID=$UID -t gluonai/gluon-cv:gpu-latest .
   ```
   
    制作docker镜像时，dockerfile中修改mxnet为1.6.0版本

- 启动Docker

   ```bash
   # 假设coco数据放在<path to coco data>目录下
   nvidia-docker run --rm -it --shm-size=8gb -v <path to coco data>:/work --net=host --privileged --name=gluon-cv gluonai/gluon-cv:gpu-latest


## 二、数据准备

YOLOv3下载[COCO2017](http://cocodataset.org/#download)数据集，解压后数据结构如下

```
dataset/coco/
  train2017/
  val2017/
  annotations
```

```bash
将coco数据集放在指定路径下
mkdir -p ~/.mxnet/datasets
ln -s <path to coco> ~/.mxnet/datasets/coco
```

## 三、测试步骤

### 1.单卡吞吐测试

对于单卡性能测试，本报告严格按Gluon-CV文档规范，对其提供的代码未做改动，由于Gluon-CV配置文件按照八卡配置，单卡测试按比例调整`batch_size`。

- 创建log文件夹并执行如下测试命令

   ```bash
   mkdir log
   export CUDA_VISIBLE_DEVICES=0
   export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
   export MXNET_EXEC_ENABLE_ADDTO=1
   python3.7 ./scripts/detection/yolo/train_yolo3.py \
         --network darknet53 \
         --dataset=coco \
         --batch-size=8 \
         --gpus=0 \
         --num-workers 8 \
         --log-interval 10 \
         --lr-decay-epoch 220,250 \
         --epochs 1 \
         --warmup-epochs 2 \
         --mixup \
         --no-mixup-epochs 20 \
         --label-smooth --no-wd \
         --save-prefix log/
   ```

- 执行后将得到如下日志文件：

   ```  
   log/yolo3_darknet53_coco_train.log 
   ```

由于模型单卡训练需要较长时间，因此设置训练1个epoch，日志仅参考模型吞吐性能，精度无需参考


### 2.八卡Time2Train吞吐测试


对于八卡性能测试，本报告严格按Gluon-CV文档规范，对其提供的代码未做改动，严格按照Gluon-CV配置进行测试。
本次测试使用416尺度进行测试, 由于Gluon-CV不统一，本地测试精度为34.8，稍低于Gluon-CV官方提供精度。

- 创建log文件夹并执行如下测试命令

   ```bash
   mkdir log
   export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
   export MXNET_EXEC_ENABLE_ADDTO=1
   python ./scripts/detection/yolo/train_yolo3.py \
         --network darknet53 \
         --dataset=coco \
         --gpus=0,1,2,3,4,5,6,7 \
         --num-workers 8 \
         --log-interval 10 \
         --lr-decay-epoch 220,250 \
         --epochs 280 \
         --warmup-epochs 2 \
         --mixup \
         --no-mixup-epochs 20 \
         --label-smooth --no-wd \
         --save-interval 1 \
         --val-interval 1 \
         --syncbn \
         --save-prefix log/
   ```

- 执行后将得到如下日志文件：

   ```
   log/yolo3_darknet53_coco_train.log 
   ```



## 四、日志数据

- [单卡吞吐测试日志](log/GPUx1_time2train_ips.log)
- [八卡Time2Train吞吐测试](log/GPUx8_time2train_ips.log) 


## 五、性能指标

|卡数 | Time2Train （sec） | 吞吐(images/sec) | 精度 |
|:-----:|:-----:|:-----:| :-----:|
|1 | - | 38.55  | - |
|8 | 468765.45 | 110.4 | 34.8 |



