# NGC PyTorch ResNet50V1.5 性能测试

此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) 实现的 ResNet50V1.5 任务的详细复现流程，包括环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [NGC PyTorch ResNet50V1.5 性能测试](#ngc-pytorch-resnet50v15-性能测试)
  - [一、环境搭建](#一环境搭建)
  - [二、数据准备](#二数据准备)
  - [三、测试步骤](#三测试步骤)
    - [1.单卡Time2Train及吞吐测试](#1单卡time2train及吞吐测试)
    - [2.单卡准确率测试](#2单卡准确率测试)
    - [3.多机Time2Train及吞吐测试](#3多机time2train及吞吐测试)
    - [4.多机准确率测试](#4多机准确率测试)
  - [四、日志数据](#四日志数据)
  - [五、性能指标](#五性能指标)

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
## 二、数据准备

下载ImageNet数据集`ISLVRC2012`,通过NGC提供的如下脚本完成解压：
```
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

## 三、测试步骤

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

### 3.多机time2train及吞吐测试
基础配置和上文所述的单机配置相同，多机这部分主要侧重于多机和单机的差异部分。

为了方便测试，我们封装了一下NGC的启动脚本

```
#!/bin/bash
set -xe

batch_size=$1  # batch size per gpu
num_gpus=$2    # number of gpu
precision=$3   # --amp or ""
train_steps=${4:-100}    # max train steps

export NODE_RANK=`python get_mpi_rank.py`

export env_path=/workspace/DeepLearningExamples/PyTorch/Classification/ConvNets
cd ${env_path}

python ./multiproc.py \
   --master_addr ${MASTER_NODE} \
   --master_port ${MASTER_PORT} \
   --nnodes ${NUM_NODES}  \
   --nproc_per_node ${num_gpus} \
   --node_rank ${NODE_RANK} \
./main.py --arch resnet50 \
	${precision} -b ${batch_size} \
	--training-only \
	-p 1 \
	--raport-file benchmark.json \
	--epochs 1 \
	--prof ${train_steps} ./data/imagenet
```

然后使用一个脚本测试多组实验

```
# fp32
echo "begin run 128 fp32 on 8 gpus"
$mpirun bash ./run_benchmark.sh  128 8 ""

echo "begin run 256 fp32 on 8 gpus"
$mpirun bash ./run_benchmark.sh  256 8 ""

# fp16
echo "begin run 128 fp16 on 8 gpus"
$mpirun bash ./run_benchmark.sh  128 8 "--amp"

echo "begin run 256 fp16 on 8 gpus"
$mpirun bash ./run_benchmark.sh  256 8 "--amp"
```

其中mpi的使用参考[这里](../../../../../../../utils/mpi.md#需要把集群节点环境传给通信框架) 

### 4.多机准确率测试

## 四、日志数据
- [单卡Time2Train及吞吐测试日志](../log/GPUx1_time2train_ips.log)
- [单卡准确率测试](../log/GPUx1_accuracy.log)
- [多机Time2Train及吞吐测试日志](./logs/32gpu_time2train_ips.log)
- [多机准确率测试](./logs/32gpu_accuracy.log

通过以上日志分析，PyTorch经过137,335秒的训练完成了90个epoch的训练，训练精度（即`val.top1`)达到76.63 %，训练吞吐（即`train.compute_ips`）达到859.24img/s。


## 五、性能指标


|              | Time2train(sec)  | 吞吐(images/sec) | 准确率(%) | 加速比 |
|--------------|------------|------------|------------|-----------|
| 1卡          |  137,335   |   859.24   |     -      |     -     |
| 8卡          |     -      |      -     |     -      |     -     |
| 32卡         |     -      |      -     |     -      |     -     |