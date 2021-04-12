<!-- omit in toc -->
# NGC PyTorch TSM 性能复现


此处给出了[TSM](https://github.com/mit-han-lab/temporal-shift-module)性能和精度测试任务的详细流程，包括执行环境、PyTorch版本、环境搭建、测试脚本、测试结果和测试日志。

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
  - [1. 添加AMP和多机代码](#1-添加AMP和多机代码)
  - [2. 单机（单卡、8卡）测试](#2-单机单卡8卡测试)
  - [3. 多机（32卡）测试](#3-多机32卡测试)
- [四、测试结果](#四测试结果)
- [五、日志数据](#五日志数据)
  - [1. 4机32卡日志](#1日志)


## 一、环境介绍

### 1.物理机环境

我们使用了同一个物理机环境，对 [TSM PyTorch](https://github.com/mit-han-lab/temporal-shift-module) 模型进行了测试,具体参数如下：
  - 系统：CentOS release 6.3 (Final)
  - GPU：Tesla V100-SXM2-16GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

本次测试所以用的docker镜像相关信息如下所示：

- **镜像版本**: `registry.baidu.com/paddlepaddle-public/paddle_ubuntu1604:mlperf_cuda10.1_cudnn7.6.5_nccl2.4.7_dali0.24.0_py37`
- **PyTorch 版本**: `1.7.0`
- **CUDA 版本**: `10.1`
- **cuDnn 版本**: `7.6.5`

## 二、环境搭建

### 1. 单机（单卡、8卡）环境搭建

单机环境搭建主要过程如下:
- 新建docker container:
使用`registry.baidu.com/paddlepaddle-public/paddle_ubuntu1604:mlperf_cuda10.1_cudnn7.6.5_nccl2.4.7_dali0.24.0_py37`docker镜像创建docker容器  

- 参考[TSM pytorch实现](https://github.com/mit-han-lab/temporal-shift-module#prerequisites) 安装依赖
    ```bash
    pip3 install torch==1.7.0 torchvision==0.8.0
    pip3 install TensorboardX
    pip3 install tqdm
    ```

- **拉取代码**

    ```bash
    git clone https://github.com/mit-han-lab/temporal-shift-module.git
    ```

### 2. 多机（32卡）环境搭建

- IB配置(可选）
请参考[这里](https://github.com/PaddlePaddle/Perf/blob/master/utils/ib.md)

- MPI配置
请参考[这里](https://github.com/PaddlePaddle/Perf/blob/master/utils/mpi.md)

### 3. 数据说明
# Kinetics-400 数据准备

- [数据集介绍](#数据集介绍)
- [下载video数据](#下载video数据)
- [提取frames数据](#提取frames数据)
- [数据预处理逻辑](#数据预处理逻辑)
- [模型说明](#模型说明)

---


## 数据集介绍

Kinetics-400是视频领域benchmark常用数据集，详细介绍可以参考其官方网站[Kinetics](https://deepmind.com/research/open-source/kinetics)。下载方式可参考官方地址[ActivityNet](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)，使用其提供的下载脚本下载数据集。

## 下载video数据

考虑到K400数据集下载困难的问题，我们将mp4格式的视频数据以zip包的形式上传到了百度云，下载方式如下：

- 下载[train_link.list](https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list)和[val_link.list](https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list)文件链接

编写下载脚本`download.sh`如下:
```bash
file=$1

while read line 
do
  wget "$line"
done <$file
```

下载训练集命令：
```bash
bash download.sh train_link.list
```

下载验证集命令:
```bash
bash download.sh train_link.list
```

|类别 | 数据条数  | list文件 |
| :------: | :----------: | :----: |
|训练集 | 234619  |  [train.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train.list)|
|验证集 | 19761 |  [val.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list)|


- 由于部分视频原始链接失效，数据有部分缺失，全部文件大概需要135G左右的存储空间，PaddleVideo使用的也是这份数据。

- 训练实用训练集数据，验证和测试均实用验证集数据，训练时数据顺序随机，测试时数据按list顺序。


## 提取frames数据
为了加速网络的训练过程，我们首先对视频文件（K400视频文件为mp4格式）提取帧 (frames)。相对于直接通过视频文件进行网络训练的方式，frames的方式能够极大加快网络训练的速度。

输入如下命令，即可提取K400视频文件的frames

```python
python extract_rawframes.py ./videos/ ./rawframes/ --level 2 --ext mp4
```

视频文件frames提取完成后，会存储在指定的`./rawframes`路径下，大小约为2T左右。

|类别 | 数据条数  | list文件 |
| :------: | :----------: | :----: |
|训练集 | 234619  |  [train_frames.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train_frames.list)|
|验证集 | 19761 |  [val_frames.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val_frames.list)|


## 数据预处理逻辑
### 训练
- GroupMultiScaleCrop: 多尺度裁剪，以[1, .875, .75, .66]比例缩放至不同的尺寸后，从固定的几个位置随机选取一个位置进行裁剪，输出尺寸为224；
- GroupRandomHorizontalFlip: 以0.5的概率随机水平翻转；
- GroupNormalize: 归一化，均值为[0.485, 0.456, 0.406]，方差为[0.229, 0.224, 0.225]。

### 测试
- GroupScale: 缩放至256大小；
- GroupCenterCrop: 中心裁剪至224大小；
- GroupNormalize: 归一化，均值为[0.485, 0.456, 0.406]，方差为[0.229, 0.224, 0.225]。


## 模型说明

### 初始化
使用ImageNet预训练模型进行参数初始化，fc层的weight实用均值为0，方差为0.001的正态分布初始化，bias初始化为0。

### 模型结构
ResNet50，bottleneck第一个conv层之前添加temporal_shift操作

### 优化器
| 优化器 | Momentum |
| :--: | :--: |
|momentum | 0.9 |
|weight_decay | 1e-4 | 
|epoch | 50 |
|总bs | 128 |
|lr | 0.02 |
|lr_decay_ratio | 0.1 |
|lr_decay_step | 20 40 |

### Loss函数
cross entropy损失

### 精度评估方法和频率
top1和top5，每训练完一次进行评估


## 三、测试步骤

### 1. 添加AMP和多机代码
由于[TSM pytorch实现](https://github.com/mit-han-lab/temporal-shift-module)没有支持AMP训练和多机训练，为了测试TSM在单机和多机上，FP32和AMP的性能和精度的具体表现情况，我们在[TSM pytorch实现](https://github.com/mit-han-lab/temporal-shift-module)做了一些修改,主要修改如下。
#### (1)为代码添加AMP支持
- 我们在[opts.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/opts.py)的末尾添加了如下一行代码，使运行时可以自由切换FP32方式或者AMP方式。
   ```bash
   parser.add_argument('--amp',default=False,action="store_true",help="use amp training")
   ```
- 我们在[main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py)代码中，依据pytorch官网[typical-mixed-precision-training](https://pytorch.org/docs/master/notes/amp_examples.html#typical-mixed-precision-training)  提供的示例参考，在[main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py)中修改一些代码。主要将GradScaler和autocast在代码中使用起来，大致修改如下，详细描述可参考此处的[main.py](https://github.com/wuhuachaocoding/AI-Rank/blob/tsm/result_v0.1/software/train/pytorch_samples/linux/tsm/code/temporal-shift-module/main.py)。
   ```bash
      scaler = torch.cuda.amp.GradScaler(args.amp)
      
      ......
      
      if use_amp:
          # compute output
          with torch.cuda.amp.autocast(enabled=use_amp):
              output = model(input_var)
              loss = criterion(output, target_var)
      else:
          output = model(input_var)
          loss = criterion(output, target_var)
      
      ......
      
      if use_amp:
          scaler.scale(loss).backward()
      else:
          loss.backward()
   
      ......
   
      if use_amp:
          scaler.unscale_(optimizer)
      
      ......
      
      if use_amp:
          scaler.step(optimizer)
          scaler.update()
      else:
          optimizer.step()
   ```  
#### (2)为代码添加多机支持   
- 为了添加多机支持，我们将[main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py)代码进行略微修改，主要包括使用DistributedDataParallel，Dataloader部分使用DistributedSampler，以及初始化进程通信相关环境。详细情况可参考此处的[main.py](https://github.com/wuhuachaocoding/AI-Rank/blob/tsm/result_v0.1/software/train/pytorch_samples/linux/tsm/code/temporal-shift-module/main.py)。
   
   
**重要的配置参数：**

- **num_trainers**: 机器数目
- **num_cards**: 每台机器使用的GPU卡数。
- **use_amp**: 是否使用AMP。

### 2. 单机（单卡、8卡）测试

为了更方便地测试单机FP32和AMP下的性能和精度，我们编写了 `run_single.sh` 脚本:

``` bash
num_trainers=$1
num_cards=$2
use_amp=$3

base=0.0025
lr_tmp=`echo "${num_cards} * ${base}" |bc`
lr="0${lr_tmp}"

if [[ ${use_amp} == True ]];then
    appends='--amp'
else
    appends=""
fi

if [[ ${num_cards} == 1 ]];then
    print_freq=32
else
    print_freq=4
fi

cd temporal-shift-module

python3 -m torch.distributed.launch --nproc_per_node ${num_cards}  main.py kinetics RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr ${lr} --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 16 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb \
     -p ${print_freq} ${appends}
```


- **单卡启动脚本：**

    若测试单机1卡AMP和FP32的训练性能和精度，执行如下命令：

    ```bash
       bash run_single.sh 1 1 True
       bash run_single.sh 1 1 False
    ```

- **8卡启动脚本：**

    若测试单机8卡AMP和FP32的训练性能和精度，执行如下命令：

    ```bash
       bash run_single.sh 1 8 True
       bash run_single.sh 1 8 False
    ```


### 3. 多机（32卡）测试
基础配置和上文所述的单机配置相同，多机这部分主要侧重于多机和单机的差异部分。
我们需要把环境变量`${MASTER_ADDR}`  `${MASTER_PORT}` `${OMPI_COMM_WORLD_RANK}`写入到`run_benchmark.sh`脚本，即可在单机的基础上完成多机的启动。

- **多机启动脚本**

	`$mpirun`命令请参考[这里](https://github.com/PaddlePaddle/Perf/blob/master/utils/mpi.md#需要把集群节点环境传给通信框架)

	```bash
	
    # AMP
    	$mpirun bash run_benchmark.sh 4 8 True
 
    # FP32
    	$mpirun bash run_benchmark.sh 4 8 False
	
	```

## 四、测试结果
(由于单卡运行时间周期过长，此处只给出单卡训练2个epoch的日志数据)
- AMP结果

|  训练卡数   | Time2train(sec)  |  准确率(%) |
|------------|------------|------------|
|    1卡     |     -      |    -     |
|    8卡     | 23711.15   |     70.121     |
|    32卡    | 10950.16   |    70.132     |
  
- FP32结果

|  训练卡数   | Time2train(sec)  |  准确率(%) |
|------------|------------|------------|
|    1卡     |   -        |    -     |
|    8卡     | 23711.15   |  70.121    |
|    32卡    | 16494.37   |  70.116    |

## 五、日志数据
### 1.日志
- [4机32卡、AMP ](../log/GPUx32_AMP.log)
- [1机8卡、AMP ](../log/GPUx8_AMP.log)
- [1机1卡、AMP ](../log/GPUx1_AMP.log)
- [4机32卡、FP32 ](../log/GPUx32_FP32.log)
- [1机8卡、FP32 ](../log/GPUx8_FP32.log)
- [1机1卡、FP32 ](../log/GPUx1_FP32.log)
