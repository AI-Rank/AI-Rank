<!-- omit in toc -->
# 基于NGC PyTorch 的mmsegmentation 性能复现

此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation) 实现的 DeepLabV3+ Pre-Training 任务的详细复现流程，包括执行环境、PyTorch版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->

## 目录
- [一、环境搭建](#一环境搭建)
  - [1. 单机8卡环境搭建](#1单机8卡环境搭建)
  - [2. python环境准备](#2python环境准备)
- [二、cityscapes数据集准备](#二cityscapes数据集准备)
- [三、测试步骤](#三测试步骤)
  - [1. 启动脚本](#1-启动脚本)
    - [1. 单卡启动](#1-单卡启动)
    - [2. 多卡启动](#2-多卡启动)
- [四、日志数据](#四日志数据)
- [五、测试结果](#五测试结果)


## 一、环境搭建  

### 1.单机8卡环境搭建
>4个节点环境一样    

我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide) 教程搭建了测试环境，主要过程如下：

- **拉取镜像**

```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
    # 本次测试是在如下版本下完成的：
    git checkout 99b1c898cead5603c945721162270c2fe077b4a2
```

- **构建镜像**

```bash
    bash scripts/docker/build.sh   # 构建镜像
```

- **从[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)拉取模型代码**

```bash
    cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
    git clone https://github.com/open-mmlab/mmsegmentation.git
    cd mmsegmentation
    # 本次测试是在如下版本下完成的：
    git checkout d0a71c1509c2d9f4cecfd775c3fb0c1e625a2d38
```


- **启动镜像**
```bash
    bash scripts/docker/launch.sh  # 启动容器
```
    我们将 `launch.sh` 脚本中的 `docker` 命令换为了 `nvidia-docker` 启动的支持 GPU 的容器，同时将`BERT`(即`$pwd`)目录替换为`mmsegmentation`目录，其他均保持不变，脚本如下：
```bash
    #!/bin/bash

    CMD=${1:-/bin/bash}
    NV_VISIBLE_DEVICES=${2:-"all"}
    DOCKER_BRIDGE=${3:-"host"}

    nvidia-docker run --name test_bert_torch -it  \
    --net=$DOCKER_BRIDGE \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e LD_LIBRARY_PATH='/workspace/install/lib/' \
    -v $PWD/mmsegmentation:/workspace/mmsegmentation \
    -v $PWD/mmsegmentation/results:/results \
    mmsegmentation $CMD
```  

### 2.python环境准备  

所需的python环境可在启动的docker容器中根据[mmsegmentation的官方文档](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation)进行配置  
>注意需要安装mmcv-full
## 二、cityscapes数据集准备  

数据下载后的结构是open-mmlab官方显示的[文件树结构](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md)，只使用了其中cityscapes的部分。  
>由于数据集比较大，且容易受网速的影响，为了更方便复现竞品的性能数据，我们使用了多进程下载方式，将原文件压缩包分为若干小包，下载之后再合并为整个包cityscapes.tar，然后进行解压

首次下载cityscapes数据时，可执行如下命令 
```bash
    # 下载cityscapes  
    wget https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar  

    # 解压数据集
    tar -xzvf cityscapes.tar

    # checksum
    md5sum cityscapes.tar
    输出：cityscapes.tar 37724b19b6e5d41f9f147936d60b3c29

    # 放到 data/ 目录下
    mv cityscapes mmsegmentation/data/
```

## 三、测试步骤

为了更准确的测试 NGC PyTorch 在 `NVIDIA DGX-1 (8x V100 32GB)` 上的性能数据，我们严格按照官方提供的模型代码配置、启动脚本，进行了性能测试。

**重要的配置参数：**  
我们测试了GPU数目与AMP对精度和性能的影响，  
GPU数目分为`单机单卡`，`单机8卡`，`4机32卡`  
AMP分为是否开启AMP，开启AMP即`fp16`，不开启即`fp32`
 
代码提交到了[mmsegmentation](./mmsegmentation)中
### 1. 启动脚本  
在自己指定好训练所需配置之后，我们就可以进行接下来的启动训练了。
#### (1) 单卡启动  
 对于在单卡上的训练，我们的启动方式是根据[官方方式](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/train.md#train-with-a-single-gpu)启动的。

```bash  
cd mmsegmentation
./run.sh > GPUx1.log 2>&1 &
```

#### (2) 多卡启动  

对于在多卡的训练(包括单机多卡和多机多卡)，我们没有使用slurm，而是统一根据官方给的[单机多卡](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/train.md#train-with-multiple-gpus)以及[多机多卡](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/train.md#train-with-multiple-machines)的第二种方法[Pytorch Lancher Utility](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py)

每轮训练在开始时速度不太稳定，而多卡的时候训练得比较块，为了可以打印出比较稳定的训练速度信息，我们在多卡时打印log会更加频繁一些，因此在多卡训练之前，我们还需要配置一下打印log的时间间隔。
- **单机8卡**  
启动命令：
``` bash
cd mmsegmentation
./run_c8.sh > GPUx8.log 2>&1 &
```  

- **多机多卡**  
启动命令与单机多卡类似，但是需要指定--nnodes, --node_rank, --master_addr, --master_port等选项，且需要在每个节点都启动，具体见[PyTorch launch utility](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py)。各节点启动训练命令如下：
>需要先修改run_c32.sh，配置好相关的参数
``` bash
cd mmsegmentation
./run_c32.sh > GPUx32.log 2>&1 &
```

## 四、日志数据  
>该模型训练是按照iter数训练的，每隔8000个iter打印一次log，因此在打印AI-Rank-Log的时候，我们将iter数映射为epoch数目

- [单卡 samples_per_gpu=2]
- [8卡 samples_per_gpu=2]
- [16卡 samples_per_gpu=2]
- [32卡 samples_per_gpu=2]

- [单卡 samples_per_gpu=4、AMP speed](../log/GPUx1.log)
- [8卡 samples_per_gpu=4、AMP train](../log/GPUx8.log)
- [16卡 samples_per_gpu=4、AMP]
- [32卡 samples_per_gpu=4、AMP train](../log/GPUx32.log)


## 五、测试结果

#### fp32测试结果

|卡数 | Time2Train(sec) | 吞吐(samples/sec) |准确率(%) | 加速比|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1 | - | 4.073 | - | - |
|8 | - | 29.74 | - | 7.3 |
|16| - |49.23  | - | 12.1 |
|32| - |81.01  | - | 19.9 |
#### fp16测试结果
|卡数 | Time2Train(sec) | 吞吐(samples/sec) |准确率(%) | 加速比|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1 | - | 8.33 | - | - |
|8 | 44338.456 | 58.2 | 78.5 | 6.98 |
|16| - |103.2 | - | 12.36 |
|32| - |182.07 | 77.3 | 21.86 |
