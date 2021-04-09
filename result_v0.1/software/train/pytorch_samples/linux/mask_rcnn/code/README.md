# Detectron Mask RCNN 性能测试

此处给出了基于 [Detectron2](https://github.com/facebookresearch/detectron2) 实现的 Mask RCNN 任务的详细复现流程，包括环境介绍、环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [Detectron2 Mask RCNN 性能测试](#detectron2-mask-rcnn-性能测试)
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

单机环境的搭建，我们遵循了 Detectron2 官网提供的 [Install](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) 和[Docker](https://github.com/facebookresearch/detectron2/tree/master/docker)教程成功搭建了测试环境，主要过程如下：


- 下载Detectron2 repo,并进入目录

   ```bash
   git clone https://github.com/facebookresearch/detectron2.git
   cd detectron2
   # 本次测试是在如下版本下完成的：
   git checkout d4412c7070b28e50037b3797de8a579afd008b2b
   ```

- 制作Docker镜像

   ```bash
   cd docker/
   
   docker build --build-arg USER_ID=$UID -t detectron2:v0 .
   ```

- 启动Docker

   ```bash
   # 假设coco数据放在<path to coco data>目录下
   nvidia-docker run --rm -it --shm-size=8gb -v <path to coco data>:/work --net=host --privileged --name=detectron2 detectron2:v0
   ```



## 二、数据准备

Mask RCNN下载[COCO2017](http://cocodataset.org/#download)数据集，解压后数据结构如下

```
dataset/coco/
  train2017/
  val2017/
  annotations
```

然后将coco文件夹上级路径加入到`DETECTRON2_DATASETS`环境变量中

```
export DETECTRON2_DATASETS=<path to dataset>
```

## 三、测试步骤

### 1.单卡吞吐测试


对于单卡性能测试，本报告严格按detectron2文档规范，对其提供的代码未做改动，由于detectron2配置文件按照八卡配置，单卡测试按比例调整`batch_size`和学习率。

- 使用如下脚本进行单卡性能测试

   ```bash
   cd tools
   CUDA_VISIBLE_DEVICES=0 ./train_net.py --num-gpus 1  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml OUTPUT_DIR ./detectron2_1gpu_fp32_bs2  SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.00125 SOLVER.MAX_ITER 5000
   ```

- 执行后将得到如下日志文件：

   ```
   
   detectron2_1gpu_fp32_bs2/log.txt 
   ```

由于模型单卡训练需要较长时间，因此设置训练轮数5000轮，日志仅进行模型吞吐，精度无需参考

### 2.八卡Time2Train吞吐测试


对于八卡性能测试，本报告严格按detectron2文档规范，对其提供的代码未做改动，严格按照detectron2配置进行测试。

- 使用如下脚本进行单卡性能测试

   ```bash
   cd tools
   CCUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./train_net.py --num-gpus 8 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml OUTPUT_DIR ./detectron2_8gpu_fp32_bs2 TEST.EVAL_PERIOD 7500
   ```

- 执行后将得到如下日志文件：

   ```
   
   detectron2_8gpu_fp32_bs2/log.txt 
   ```
detectron2 Mask RCNN模型总训练轮数为9000轮，等价于12个epoch，因此间隔7500个iter，等价于1个epoch进行评估


## 四、日志数据


- [单卡吞吐测试日志](log/GPUx1_time2train_ips.log) 
- [八卡Time2Train吞吐测试](log/GPUx8_time2train_ips.log) 


## 五、性能指标

|卡数 | Time2Train （sec） | 吞吐(images/sec) | 精度 |
|:-----:|:-----:|:-----:| :-----:|
|1 | - | 9.66  | - |
|8 | 22898 | 55.26 | box ap: 37.8 mask ap 34.4 |

