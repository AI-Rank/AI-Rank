<!-- omit in toc -->
# NGC PyTorch Bert 性能复现


此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) 实现的 Bert Base Pre-Training 任务的详细复现流程，包括执行环境、PyTorch版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->
## 目录
- [一、环境搭建](#一环境搭建)
- [二、Bert wiki-only 数据集的准备](#二bert-wiki-only-数据集的准备)
- [三、测试步骤](#三测试步骤)
  - [1. 单机吞吐测试](#1-单机吞吐测试)
- [四、日志数据](#四日志数据)
- [五、性能数据](#五性能数据)



## 一、环境搭建

我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide) 教程搭建了测试环境，主要过程如下：

- **拉取代码**

    ```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
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

    CMD=${1:-/bin/bash}
    NV_VISIBLE_DEVICES=${2:-"all"}
    DOCKER_BRIDGE=${3:-"host"}

    nvidia-docker run --name test_bert_torch -it  \
    --net=$DOCKER_BRIDGE \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e LD_LIBRARY_PATH='/workspace/install/lib/' \
    -v $PWD:/workspace/bert \
    -v $PWD/results:/results \
    bert $CMD
    ```

- **准备数据**

    NGC PyTorch 提供单独的数据下载和预处理脚本，详细的数据处理流程请参考[此处](../data/README.md)。

## 二、Bert wiki-only 数据集的准备

首先根据 NGC PyTorch 的代码仓库提供的自动构建 Docker 镜像的的 [shell 脚本](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT/scripts/docker/build.sh)创建容器，可参考[此处](../script/README.md) 。


- **数据下载**

    NGC PyTorch 提供单独的数据下载和预处理脚本 [data/create_datasets_from_start.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/data/create_datasets_from_start.sh)。在容器中执行如下命令，可以下载和制作 `wikicorpus_en` 的 hdf5 数据集。

    ```bash
    bash data/create_datasets_from_start.sh wiki_only
    ```

    由于数据集比较大，且容易受网速的影响，上述命令执行时间较长。因此，为了更方便复现竞品的性能数据，我们提供了已经处理好的 seq_len=128 的 hdf5 格式[样本数据集](https://bert-data.bj.bcebos.com/benchmark_sample%2Fhdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar.gz)，共100个 part hdf5 数据文件，约 3.1G。

    数据下载后，会得到一个 `hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar.gz`压缩文件：

    ```bash
    # 解压数据集
    tar -xzvf benchmark_sample_hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.tar.gz

    # 放到 data/ 目录下
    mv benchmark_sample_hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5 bert/data/
    ```

    修改 [scripts/run_pretraining.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/scripts/run_pretraining.sh#L37)脚本的 `DATASET`变量为上述数据集地址即可。


## 三、测试步骤

为了更准确的测试 NGC PyTorch 在 `NVIDIA DGX-1 (8x V100 16GB)` 上的性能数据，我们严格按照官方提供的模型代码配置、启动脚本，进行了的性能测试。

官方提供的 [scripts/run_pretraining.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/scripts/run_pretraining.sh) 执行脚本中，默认配置的是两阶段训练。我们此处统一仅执行 **第一阶段训练**，并根据日志中的输出的数据计算吞吐。

**重要的配置参数：**

- **train_batch_size**: 用于第一阶段的单卡总 batch_size, 单卡每步有效 `batch_size = train_batch_size / gradient_accumulation_steps`
- **precision**: 用于指定精度训练模式，fp32 或 fp16
- **use_xla**: 是否开启 XLA 加速，我们统一开启此选项
- **num_gpus**: 用于指定 GPU 卡数
- **gradient_accumulation_steps**: 每次执行 optimizer 前的梯度累加步数
- **BERT_CONFIG:** 用于指定 base 或 large 模型的参数配置文件 (line:49)
- **bert_model:** 用于指定模型类型，默认为`bert-large-uncased`

### 1. 单机吞吐测试

由于官方默认给出的是支持两阶段训练的 **Bert Large** 模型的训练配置，若要测**Bert Base**模型，需要对 `run_pretraining.sh` 进行如下改动：

- 在 `bert` 项目根目录新建一个 `bert_config_base.json` 配置文件，内容如下：

  ```
  {
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 30522
  }
  ```

- 修改 `run_pretraining.sh`的第39行内容为：

  ```bash
  BERT_CONFIG=bert_config_base.json
  ```
- 修改 `run_pretraining.sh`的第110行内容为：

  ```bash
  CMD+=" --bert_model=bert-base-uncased"
  ```
- 由于不需要执行第二阶段训练，故需要注释 `run_pretraining.sh` 的第154行到最后，即：

  ```bash
  #Start Phase2

  # PREC=""
  # if [ "$precision" = "fp16" ] ; then
  #    PREC="--fp16"

  # ......(此处省略中间部分)

  # echo "finished phase2"
  ```

同时，为了更方便地测试不同 batch_size、num_gpus、precision组合下的 Pre-Training 性能，我们单独编写了 `run_benchmark.sh` 脚本，并放在`scripts`目录下。

- **shell 脚本内容如下：**

    ```bash
    #!/bin/bash

    set -x

    batch_size=$1  # batch size per gpu
    num_gpus=$2    # number of gpu
    precision=$3   # fp32 | fp16
    gradient_accumulation_steps=$(expr 67584 \/ $batch_size \/ $num_gpus)
    train_batch_size=$(expr 67584 \/ $num_gpus)   # total batch_size per gpu
    train_steps=${4:-250}    # max train steps

    # NODE_RANK主要用于多机，单机可以不用这行。
    export NODE_RANK=`python get_mpi_rank.py`
    # 防止checkpoints冲突
    rm -rf results/checkpoints

    # run pre-training
    bash scripts/run_pretraining.sh $train_batch_size 6e-3 $precision $num_gpus 0.2843 $train_steps 200 false true true $gradient_accumulation_steps
    ```
    > 注：由于原始 global_batch_size=65536 对于 batch_size=48/96 时出现除不尽情况。因此我们按照就近原则，选取 67584 作为 global_batch_size.<br>
    > 计算公式：global_batch_size = batch_size_per_gpu * num_gpu * num_accumulation_steps

- **单卡启动脚本：**

    若测试单机单卡 batch_size=48、AMP 的训练性能，执行如下命令：

    ```bash
    bash scripts/run_benchmark.sh 48 1 AMP
    ```


## 四、日志数据

- [单机吞吐日志](../logs/bert_base_lamb_pretraining_phase1_fp16_bs96_gpu1.log)

通过以上日志分析，PyTorch 在 Bert Pre-training 任务上的单机吞吐达到了 **543.76** `samples/sec` 。

> 注：
> 1. 由于 Bert 的训练数据集非常大，需要多机多卡进行训练。因资源有限，此处未给出单机训练的 Time2Train数据。

## 五、性能数据

|              | Time2train(sec)  | 吞吐(samples/sec) | 准确率(%) | 加速比 |
|--------------|------------|------------|------------|-----------|
| 1卡          |     -      |   543.76   |     -      |     -     |
| 8卡          |     -      |      -     |     -      |     -     |
| 32卡         |     -      |      -     |     -      |     -     |