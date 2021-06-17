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
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 48
  - Driver Version: 450.80.02
  - 内存：502 GB

### 2.Docker 镜像

本次测试所以用的docker镜像相关信息如下所示：

- **镜像版本**: `registry.baidu.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.0-cudnn8`
- **PyTorch 版本**: `1.7.0`
- **CUDA 版本**: `11.0`
- **cuDnn 版本**: `8.0.4`

## 二、环境搭建

### 1. 单机（单卡、8卡）环境搭建

单机环境搭建主要过程如下:
- 新建docker container:
使用`registry.baidu.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.0-cudnn8`docker镜像创建docker容器  

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
- 具体数据使用情况，请参看[DATA.md](DATA.md)

## 三、测试步骤

### 1. 添加AMP和多机代码
由于[TSM pytorch实现](https://github.com/mit-han-lab/temporal-shift-module)没有支持AMP训练和多机训练，为了测试TSM在单机和多机上，FP32和AMP的性能和精度的具体表现情况，我们在[TSM pytorch实现](https://github.com/mit-han-lab/temporal-shift-module)做了一些修改,主要修改如下。
#### (1)为代码添加AMP支持
- 我们在[opts.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/opts.py)的末尾添加了如下一行代码，使运行时可以自由切换FP32方式或者AMP方式。
   ```bash
   parser.add_argument('--amp',default=False,action="store_true",help="use amp training")
   ```

- 我们在[main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py)代码中，依据pytorch官网[typical-mixed-precision-training](https://pytorch.org/docs/master/notes/amp_examples.html#typical-mixed-precision-training)  提供的示例参考，在[main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py)中修改一些代码。主要将GradScaler和autocast在代码中使用起来，详细描述可参考此处的[main.py](temporal-shift-module/main.py)。
  
#### (2)为代码添加多机支持   
- 为了添加多机支持，我们将[main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py)代码进行略微修改，主要包括使用DistributedDataParallel，Dataloader部分使用DistributedSampler，以及初始化进程通信相关环境。详细情况可参考此处的[main.py](temporal-shift-module/main.py)。

   
   
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

|  训练卡数   | Time2train(sec)  |吞吐(samples/sec)  |  准确率(%) |
|------------|------------|------------|------------|
|    1卡     |     -      |    55.365     |    -     |
|    8卡     | 29802.62   |   383.72      |     71.118     |
|    32卡    | 8130.06   |   1487.488      |    71.071     |
  

## 五、日志数据
### 1.日志
- [4机32卡、AMP ](../log/gPUx32_AMP.log)
- [1机8卡、AMP ](../log/gPUx8_AMP.log)
- [1机1卡、AMP ](../log/gPUx1_AMP.log)

