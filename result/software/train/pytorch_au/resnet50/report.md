# NGC PyTorch ResNet50V1.5 性能测试

此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) 实现的 ResNet50V1.5 任务的详细复现流程，包括环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [NGC PyTorch ResNet50V1.5 性能测试](#ngc-pytorch-resnet50v15-性能测试)
  - [一、环境搭建](#一环境搭建)
  - [二、测试步骤](#二测试步骤)
    - [1.单卡Time2Train及吞吐测试](#1单卡time2train及吞吐测试)
    - [2.单卡准确率测试](#2单卡准确率测试)
  - [三、日志数据](#三日志数据)

## 一、环境搭建

我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：


- 下载NGC PyTorch repo,并进入目录

   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples
   cd DeepLearningExamples/PyTorch/Classification/ConvNets
   # 本次测试是在如下版本下完成的：
   git checkout 99b1c898cead5603c945721162270c2fe077b4a2
   ```

- 制作Docker镜像

   ```bash
   docker build . -t nvidia_rn50_pytorch
   ```

- 启动Docker

   ```bash
   # 假设imagenet数据放在<path to data>目录下
   nvidia-docker run --rm -it -v <path to data>:/imagenet --ipc=host nvidia_rn50_pytorch
   ```

## 二、测试步骤

### 1.单卡Time2Train及吞吐测试

根据NGC公布的数据，90个epoch精度可以达到AI-Rank要求的76.10%，我们使用如下脚本测试：

```
    python ./launch.py --model resnet50 --precision AMP --mode convergence --platform DGX1V /imagenet --epochs 90 --mixup 0.0 --workspace ${1:-./} --raport-file raport.json 
```

### 2.单卡准确率测试

根据NGC公布的数据，250个epoch基本可以达到最佳训练效果，我们使用如下脚本测试：

```
    python ./multiproc.py --nproc_per_node 8 ./launch.py --model resnet50 --precision AMP --mode convergence --platform DGX1V /imagenet --workspace ${1:-./} --raport-file raport.json
```

## 三、日志数据
- [单卡Time2Train及吞吐测试日志](./logs/1gpu_time2train_ips.log)
- [单卡准确率测试](./logs/1gpu_accuracy.log)

通过以上日志分析，PyTorch经过137,335秒的训练完成了90个epoch的训练，训练精度（即`val.top1`)达到76.63 %，训练吞吐（即`train.compute_ips`）达到859.24img/s。
