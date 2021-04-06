<!-- omit in toc -->
# NGC PyTorch TSM 性能复现


此处给出了基于 [NGC PyTorch](https://github.com/mit-han-lab/temporal-shift-module) 实现的 TSM 任务的详细复现流程，包括执行环境、PyTorch版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->
## 目录
- [一、环境介绍](#一环境介绍)
  - [1.物理机环境](#1物理机环境)
  - [2.Docker 镜像](#2docker-镜像)
- [二、环境搭建](#二环境搭建)
  - [1. 单机（单卡、8卡）环境搭建](#1-单机单卡8卡环境搭建)
  - [2. 多机（32卡）环境搭建](#2-多机32卡环境搭建)
- [三、测试步骤](#三测试步骤)
  - [1. 单机（单卡、8卡）测试](#1-单机单卡8卡测试)
  - [2. 多机（32卡）测试](#2-多机32卡测试)
- [四、测试结果](#四测试结果)
- [五、日志数据](#五日志数据)
  - [1.单机（单卡、8卡）日志](#1单机单卡8卡日志)


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
**重要的配置参数：**

- **max-tokens**: 单卡 batch_size
- **amp**: 用于指定是否开启amp训练。
- **amp-devel**: O1代表amp，O2代表fp16。

### 2. 单机（单卡、8卡）测试

为了更方便地测试不同 batch_size、num_gpus、precision组合下的性能，我们编写了 `run_benchmark.sh` 脚本:

``` bash
num_trainers=${OMPI_COMM_WORLD_SIZE:-1}
num_gpu=$1
batch=$2
total_cards=$((num_trainers*num_gpu))
if [[ $3 == 'fp32' ]];then
    appends=""
elif [[ $3 == 'fp16' ]];then
    appends='--amp --amp-level O2'
elif [[ $3 == 'amp' ]];then
    appends='--amp --amp-level O1'
else
    echo "unexpect fp32 fp16 or amp"
    exit
fi

# RANK主要用于多机，单机可以不用这行
export RANK=${OMPI_COMM_WORLD_RANK:-0}

distribute="--nnodes ${num_trainers} --node_rank ${RANK}  \
    --master_addr ${MASTER_ADDR:-127.0.0.1} --master_port ${MASTER_PORT:-8421}"
envs='--distributed-init-method env://'


python  -m torch.distributed.launch --nproc_per_node=${num_gpu}  ${distribute} train.py \
  /data/wmt14_en_de_joined_dict \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000  \
  --lr 0.000846 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens ${batch} \
  --seed 1 \
  --max-epoch 40 \
  --no-epoch-checkpoints \
  --fuse-layer-norm \
  --online-eval \
  --log-interval 10 \
  --max-update $6 \
  ${envs} \
  --save-dir /workspace/checkpoint \
  ${appends} 

```


- **单卡启动脚本：**

    若测试单机单卡 batch_size=2560、FP32 的训练性能，执行如下命令：

    ```bash
    bash scripts/run_benchmark.sh 1 2560 fp32
    ```

- **8卡启动脚本：**

    若测试单机8卡 batch_size=5120、FP16 的训练性能，执行如下命令：

    ```bash
    bash scripts/run_benchmark.sh 8 5120 fp16
    ```


### 3. 多机（32卡）测试
基础配置和上文所述的单机配置相同，多机这部分主要侧重于多机和单机的差异部分。
我们需要把环境变量`${MASTER_ADDR}`  `${MASTER_PORT}`传递给`run_pretraining.sh`脚本，即可在单机的基础上完成多机的启动。

- **多机启动脚本**

	`$mpirun`命令请参考[这里](../../../utils/mpi.md#需要把集群节点环境传给通信框架)

	```
	# fp32
	echo "begin run bs:2560 fp32 on 8 gpus"
	$mpirun bash ./run_benchmark.sh 8 2560 fp32

    # amp
	echo "begin run bs:5120 amp on 8 gpus"
	$mpirun bash ./run_benchmark.sh 8 5120 amp
 
    # fp16
	echo "begin run bs:5120 fp16 on 8 gpus"
	$mpirun bash ./run_benchmark.sh 8 5120 fp16

	# add more test
	```

## 四、测试结果

> 单位： words/sec

|卡数 | FP32(BS=2560) | AMP(BS=5120) | FP16(BS=5120) |
|:-----:|:-----:|:-----:|:-----:|
|1 |  |  |  |
|8 |  |  |  |
|32 | 166352.6 | 385625.7 | 590188.7 |

## 五、日志数据
### 1.日志
- [4机32卡、FP32](./logs/pytorch_gpu32_fp32_bs2560)
- [4机32卡、FP16](./logs/pytorch_gpu32_fp16_bs5120)
- [4机32卡、AMP ](./logs/pytorch_gpu32_amp_bs5120)
