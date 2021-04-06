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

## 三、测试步骤

### 1. 添加AMP和多机代码
由于[TSM pytorch实现](https://github.com/mit-han-lab/temporal-shift-module)没有支持AMP训练和多机训练，为了测试TSM在单机和多机上，FP32和AMP的性能和精度的具体表现情况，我们在[TSM pytorch实现](https://github.com/mit-han-lab/temporal-shift-module)做了一些修改,主要修改如下。
#### (1)为代码添加AMP支持
- 我们在[opts.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/opts.py)的第77行添加了如下一行代码，使运行时可以自由切换FP32方式或者AMP方式。
   ```bash
   parser.add_argument('--amp',default=False,action="store_true",help="use amp training")
   ```
- 我们在[main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py)代码中，依据pytorch官网[typical-mixed-precision-training](https://pytorch.org/docs/master/notes/amp_examples.html#typical-mixed-precision-training)  提供的示例参考，在[main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py)中修改一些代码。在第185行添加
   ```bash
      scaler = torch.cuda.amp.GradScaler(args.amp)
   ```  
   将194行修改为
   ```bash
      train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer,scaler,args.amp,rank)
   ```
   相应的train函数定义处做相应的修改
   ```bash
      def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer,scaler,use_amp,rank=None):
   ```  
   将第243-246行之间的代码换成
   ```bash
      if use_amp:
          # compute output
          with torch.cuda.amp.autocast(enabled=use_amp):
              output = model(input_var)
              loss = criterion(output, target_var)
      else:
          output = model(input_var)
          loss = criterion(output, target_var)
   ```  
   将254行代码修改为
   ```bash
      if use_amp:
          scaler.scale(loss).backward()
      else:
          loss.backward()
   ```  
   在第257行上面添加以下代码
   ```bash
      if use_amp:
          scaler.unscale_(optimizer)
   ```  
   将第259行代码替换成以下代码
   ```bash
      if use_amp:
          scaler.step(optimizer)
          scaler.update()
      else:
          optimizer.step()
   ```  
#### (2)为代码添加多机支持   
- 为了添加多机支持，我们将[main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py)代码中的第74行修改为
   ```bash
      model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[device_id], output_device=device_id)
   ```  
   在第140行添加
   ```bash
      train_sampler = DistributedSampler(train_dataset, shuffle=True)
   ```  
   将第141-154之间的代码替换为
   ```bash
      train_loader = torch.utils.data.DataLoader(
           TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
               batch_size=args.batch_size, 
	   sampler=train_sampler,
           num_workers=args.workers, pin_memory=True,
           drop_last=True)  # prevent something not % n_GPU
   ```  
   在第192行添加
   ```bash
      train_sampler.set_epoch(epoch)
   ```  
- 添加多机通信支持。我们在[main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py)中添加一个新函数
   ```bash
      def run():
          current_env = os.environ.copy()
          current_env["MASTER_ADDR"] = os.environ.get('MASTER_ADDR', "10.225.135.14")
          current_env["MASTER_PORT"] = os.environ.get('MASTER_PORT', 29500)
          current_env["WORLD_SIZE"] = os.environ.get('WORLD_SIZE', 32)
          current_env["OMP_NUM_THREADS"] = str(1)
          distributed_init_method = r'env://'
          dist.init_process_group(backend="nccl", init_method=distributed_init_method)
          distributed_world_size = int(os.environ['WORLD_SIZE'])
          distributed_rank = dist.get_rank()
          main(distributed_rank)
   ```  
   并将第378行代码修改成
   ```bash
      run()
   ```  
   
   
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

cd /root/paddlejob/workspace/env_run/pytorch_TSM

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
(由于单卡运行时间周期过长，此处只给出单卡的性能数据)
- 性能结果
> 单位： batch/sec

|卡数 | FP32 | AMP |
|:-----:|:-----:|:-----:|
|1 |  |  |
|8 |  |  |
|32 |  | 1380.045 |
  
- 精度结果
> top1 acc

|卡数 | FP32 | AMP |
|:-----:|:-----:|:-----:|
|8 |  |  |
|32 |  | 71.199 |

## 五、日志数据
### 1.日志
- [4机32卡、AMP ](./logs/final_amp_32.txt)
