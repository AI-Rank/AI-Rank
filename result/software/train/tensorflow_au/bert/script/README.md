<!-- omit in toc -->
# NGC TensorFlow Bert 性能复现

此处给出了基于 [NGC TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) 实现的 Bert Base Pre-Training 任务的详细复现流程，包括执行环境、TensorFlow版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->
## 目录
- [一、环境介绍](#一环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [二、环境搭建](#二环境搭建)
  - [1. 单机（单卡、8卡）环境搭建](#1-单机单卡8卡环境搭建)
- [三、测试步骤](#三测试步骤)
  - [1. 单机（单卡、8卡）测试](#1-单机单卡8卡测试)
- [四、测试结果](#四测试结果)
- [五、日志数据](#五日志数据)
  - [1.单机（单卡、8卡）日志](#1单机单卡8卡日志)


## 一、环境介绍

### 1.物理机环境

我们使用了同一个物理机环境，对 [NGC TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) 的 Bert 模型进行了测试，详细物理机配置：

- 单机（单卡、8卡）
  - 系统：CentOS Linux release 7.5.1804
  - GPU：Tesla V100-SXM2-16GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 38
  - Driver Version: 450.80.02
  - 内存：432 GB

### 2.Docker 镜像

NGC TensorFlow 的代码仓库提供了自动构建 Docker 镜像的的 [shell 脚本](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/docker/build.sh)，支持一键构建和启动容器，测试环境如下：

- **镜像版本**: `nvcr.io/nvidia/tensorflow:20.06-tf1-py3`
- **TensorFlow 版本**: `1.15.2+nv`
- **CUDA 版本**: `11.0`
- **cuDnn 版本**: `8.0.1`

## 二、环境搭建

### 1. 单机（单卡、8卡）环境搭建

我们遵循了 NGC TensorFlow 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：

- **拉取代码**

  ```bash
  git clone https://github.com/NVIDIA/DeepLearningExamples
  cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT
  # 本次测试是在如下版本下完成的：
  git checkout 99b1c898cead5603c945721162270c2fe077b4a2
  ```

- **构建镜像**

  ```bash
  bash scripts/docker/build.sh   # 构建镜像
  bash scripts/docker/launch.sh  # 启动容器
  ```

  我们将 `launch.sh` 脚本中的 `docker` 命令换为了 `nvidia-docker` 启动的支持 GPU 的容器，其他均保持不变，脚本如下：
  ```bash
  #!/bin/bash

  CMD=${@:-/bin/bash}
  NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

  nvidia-docker run --name=test_tf_bert -it \
      --net=host \
      --shm-size=1g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
      -v $PWD:/workspace/bert \
      -v $PWD/results:/results \
      bert $CMD
  ```

- **准备数据**

  NGC TensorFlow 提供单独的数据下载和预处理脚本，详细的数据处理流程请参考[此处](../data/README.md)。

## 三、测试步骤

为了更准确的测试 NGC TensorFlow 在 `NVIDIA DGX-1 (8x V100 16GB)` 的性能数据，我们严格按照官方提供的模型代码配置、启动脚本，进行了的性能测试。

官方提供的 [scripts/run_pretraining_lamb.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb.sh) 执行脚本中，默认配置的是两阶段训练。我们此处统一仅执行 **第一阶段训练**，并根据日志中的输出的数据计算吞吐。因此我们注释掉了[scripts/run_pretraining_lamb.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb.sh#L60)的60行：

```bash
# RUN PHASE 2
# bash scripts/run_pretraining_lamb_phase2.sh $SCRIPT_ARGS |& tee -a $LOGFILE
```

**重要的配置参数：**

- **train_batch_size_phase1**: 用于指定每张卡上的 batch_size 数目
- **precision**: 用于指定精度训练模式，fp32 或 fp16
- **use_xla**: 是否开启 XLA 加速，我们统一开启此选项
- **num_gpus**: 用于指定 GPU 卡数
- **bert_model**: 用于指定 Bert 模型，我们统一指定为 **base**

### 1. 单机（单卡、8卡）测试

为了更方便地测试不同 batch_size、num_gpus、precision组合下的 Pre-Training 性能，我们单独编写了 `run_benchmark.sh` 脚本，并放在`scripts`目录下。

- **shell 脚本内容如下：**

  ```bash
  #!/bin/bash

  set -x

  batch_size=$1  # batch size per gpu
  num_gpus=$2    # number of gpu
  precision=$3   # fp32 | fp16
  num_accumulation_steps_phase1=$(expr 67584 \/ $batch_size \/ $num_gpus)
  train_steps=${4:-200}        # max train steps
  bert_model=${5:-"base"}      # base | large

  # run pre-training
  bash scripts/run_pretraining_lamb.sh $batch_size 64 8 7.5e-4 5e-4 $precision true $num_gpus 2000 200 $train_steps 200 $num_accumulation_steps_phase1 512 $bert_model
  ```
  > 注：由于原始 global_batch_size=65536 对于 batch_size=48/96 时出现除不尽情况。因此我们按照就近原则，选取 67584 作为 global_batch_size.<br>
  > 计算公式：global_batch_size = batch_size_per_gpu * num_gpu * num_accumulation_steps


- **单卡启动脚本：**

  若测试单机单卡 batch_size=32、FP32 的训练性能，执行如下命令：

  ```bash
  bash scripts/run_benchmark.sh 32 1 fp32
  ```

- **8卡启动脚本：**

  若测试单机8卡 batch_size=96、AMP 的训练性能，执行如下命令：

  ```bash
  bash scripts/run_benchmark.sh 96 8 fp16
  ```

## 四、测试结果


|卡数 | Time2Train(cec) | 吞吐(samples/sec) |准确率(%) | 加速比|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1 | - | 536.06 | - | - |
|8 | - | 3530.84 | - | - |


> 注：
> 1. 由于 Bert 的训练数据集非常大，需要多机多卡进行训练。因资源有限，此处未给出单机训练的 Time2Train数据。
> 2. 我们分别测试了 FP32 下 bs=32/48、以及 AMP 下 bs=64/96 性能数据，选取最优的组合 AMP(bs=96) 作为最终吞吐数据。

## 五、日志数据
### 1.单机（单卡、8卡）日志

- [单卡 bs=96、AMP](../logs/tf_bert_pretraining_lamb_base_fp16_bs96_gpu1.log)
- [8卡 bs=96、AMP](../logs/tf_bert_pretraining_lamb_base_fp16_bs96_gpu8.log)
