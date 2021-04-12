# Tensorflow Wide&Deep 性能测试

此处给出基于 `Tensorflow-1.12.0` 实现的 Wide&Deep 任务的详细复现流程，包括执行环境、Tensorflow版本、环境搭建、复现脚本、测试结果和测试日志，我们将主要测试在参数服务器模式下的性能，使用CPU集群产出模型训练的性能。

## 目录
- [Tensorflow Wide&Deep 性能测试](#tensorflow-widedeep-性能测试)
  - [目录](#目录)
  - [一、环境搭建](#一环境搭建)
  - [二、Criteo 数据集的准备](#二criteo-数据集的准备)
    - [1. 原始数据集的下载](#1-原始数据集的下载)
    - [2. 数据集预处理](#2-数据集预处理)
  - [三、测试步骤](#三测试步骤)
    - [1. 单节点吞吐测试](#1-单节点吞吐测试)
    - [2. 多节点吞吐测试](#2-多节点吞吐测试)
  - [四、日志数据](#四日志数据)
  - [五、性能指标](#五性能指标)

## 一、环境搭建

我们使用Tensorflow官方提供的[Docker](https://hub.docker.com/r/tensorflow/tensorflow/)镜像，测试环境如下：

- **镜像版本**: `tensorflow/tensorflow:1.12.0`
- **Tensorflow版本**: `1.12.0`

我们在通用K8S集群上成功搭建了测试环境，针对任务训练节点数量及参数服务器数量的配置，集群将分别启动对应数量的pod，并配置分布式训练所需超参，完成参数服务器模式的训练。

执行以下命令以下载及进入容器：

``` bash
docker pull tensorflow/tensorflow:1.12.0
docker run -it --net=host tensorflow/tensorflow:1.12.0 /bin/bash
```


## 二、Criteo 数据集的准备

训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。


此处给出本任务所需的`Criteo`数据集的下载及制作流程。

### 1. 原始数据集的下载

``` bash
# 原始数据集已提前上传至百度云BOS，国内网络可快速下载
echo "Begin DownLoad Criteo Data"
wget --no-check-certificate https://paddlerec.bj.bcebos.com/benchmark/criteo_benchmark_data.tar.gz 
echo "Begin Unzip Criteo Data"
tar -xf criteo_benchmark_data.tar.gz
echo "Get Criteo Data Success"
```

Criteo 原始数据集格式如下：
```basg
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```

其中`label`表示广告是否被点击，点击用1表示，未点击用0表示。`integer feature`代表数值特征（连续特征），共有13个连续特征。`categorical feature`代表分类特征（离散特征），共有26个离散特征。相邻两个特征用`\t`分隔，缺失特征用空格表示。

### 2. 数据集预处理

我们对原始Criteo数据集进行了以下步骤的处理:

- 数值型特征进行归一化

    我们提前对原始数据进行统计及筛选后，我们得到了每一个维度的数值型特征，95%数据的 `最小值`、`最大值`及`数值范围`，以此为标准进行预处理，避免长尾极值数据对训练产生影响，提高模型泛化性：

    ```python
    # 数值型特征每一个维度的最小值、最大值及数值范围
    cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
    cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
    # 处理公式如下
    # label: idx = 0
    # dense_feature: idx [1,14)
    new_dense_feature[idx] = float(origin_dense_feature[idx]) - cont_min_[idx-1]) / cont_diff_[idx-1])
    ```

- 稀疏离散特征hash

    我们提前对稀疏特征数据进行统计后，选择 hash_dim = 1000001 作为特征映射的基准，并在映射前，对原始稀疏特征字符串加上其下标，进一步明确该feasign所属slot的信息，以避免冲突

    ```python
    hash_dim = 1000001
    # 处理公式如下
    # label: idx = 0
    # dense_feature: idx [1,14)
    # sparse_feature: idx [14,40)
    new_sparse_feature[idx] = hash(str(idx) + origin_sparse_feature[idx]) % hash_dim_
    ```

我们将原始数据集处理后，保存到`csv`文件中，使用Tensorflow的异步数据读取模式进行训练。您可以直接下载我们预处理后的数据集：

```bash
# 经过预处理，可直接读取的数据集
echo "Begin DownLoad Criteo Data"
wget --no-check-certificate https://paddlerec.bj.bcebos.com/benchmark/tf_criteo.tar.gz
echo "Begin Unzip Criteo Data"
tar -xf tf_criteo.tar.gz
echo "Get Criteo Data Success"

```

## 三、测试步骤

Wide&Deep模型适用的场景是搜索、广告以及推荐，该领域模型训练数据量大，稀疏特征维度巨大，同时模型结构简单，通常使用参数服务器模式进行训练。我们在此主要测试模型单节点性能，以及扩展后的多节点的性能，并统计加速比，使用AUC衡量模型训练效果。

### 1. 单节点吞吐测试

为了更方便地测试不同 batch_size、num_threads、node_nums 组合下的性能，我们单独编写了 `run_single_train.sh` 脚本

```bash
#!/bin/bash

if [ ! -d "./output/checkpoint/sync" ]; then
    mkdir -p ./output/checkpoint/sync
fi

if [ ! -d "./output/checkpoint/async" ]; then
    mkdir -p ./output/checkpoint/async
fi

if [ ! -d "./log/sync" ]; then
    mkdir -p ./log/sync
fi

if [ ! -d "./log/async" ]; then
    mkdir -p ./log/async
fi

export PADDLE_PSERVERS_IP_PORT_LIST=127.0.0.1:36001
export PADDLE_WORKERS_IP_PORT_LIST=127.0.0.1:36006

trainer_nums=1
pserver_nums=1


for((i=0;i<${pserver_nums};i++)) 
do
    export TRAINING_ROLE=PSERVER
    export PADDLE_TRAINER_ID=$i
    python -u wide_deep_distribute.py --sync_mode=False  &> ./log/${mode}/pserver.$i.log &
done

for((i=0;i<${trainer_nums};i++))
do
    export TRAINING_ROLE=TRAINER
    export PADDLE_TRAINER_ID=$i
    python -u wide_deep_distribute.py --sync_mode=False &> ./log/${mode}/worker.$i.log &
done

```


### 2. 多节点吞吐测试

我们首先给出本地模拟分布式多节点训练的脚本:

```bash
#!/bin/bash

if [ ! -d "./output/checkpoint/sync" ]; then
    mkdir -p ./output/checkpoint/sync
fi

if [ ! -d "./output/checkpoint/async" ]; then
    mkdir -p ./output/checkpoint/async
fi

if [ ! -d "./log/sync" ]; then
    mkdir -p ./log/sync
fi

if [ ! -d "./log/async" ]; then
    mkdir -p ./log/async
fi

export PADDLE_PSERVERS_IP_PORT_LIST=127.0.0.1:36001,127.0.0.1:36002
export PADDLE_WORKERS_IP_PORT_LIST=127.0.0.1:36006,127.0.0.1:36007
trainer_nums=2
pserver_nums=2


for((i=0;i<${pserver_nums};i++)) 
do
    export TRAINING_ROLE=PSERVER
    export PADDLE_TRAINER_ID=$i
    python -u wide_deep_distribute.py --sync_mode=False  &> ./log/${mode}/pserver.$i.log &
done

for((i=0;i<${trainer_nums};i++))
do
    export TRAINING_ROLE=TRAINER
    export PADDLE_TRAINER_ID=$i
    python -u wide_deep_distribute.py --sync_mode=False &> ./log/${mode}/worker.$i.log &
done

```

相比单节点，会扩展`PADDLE_PSERVERS_IP_PORT_LIST`以及`PADDLE_WORKERS_IP_PORT_LIST`，同时更改`trainer_nums` 以及 `pserver_nums`配置。

在集群上运行时，按照以下步骤进行分布式训练：
1. 需要首先确定`PADDLE_PSERVERS_IP_PORT_LIST`以及`PADDLE_WORKERS_IP_PORT_LIST`，确保节点之间通过该IP:PORT可访问
2. 针对扮演参数服务器角色的节点，配置` TRAINING_ROLE=PSERVER`，同时给定其顺位`PADDLE_TRAINER_ID`
3. 针对扮演训练器角色的节点，配置`TRAINING_ROLE=TRAINER`，同时给定其顺位`PADDLE_TRAINER_ID`


## 四、日志数据

- [单节点吞吐日志]()

通过以上日志分析，Tensorflow ParameterServer模式， 单节点的吞吐为 **11072.04** `samples/sec`

- [四节点吞吐日志]()

通过以上日志分析，Tensorflow ParameterServer模式， 四节点时，单节点的吞吐为 **22162.16** `samples/sec`， 总吞吐可计算为

- [八节点吞吐日志]()

通过以上日志分析，Tensorflow ParameterServer模式， 八节点的吞吐为 **22187.04** `samples/sec`

## 五、性能指标

|       | Time2train(sec) | 吞吐(samples/sec) | AUC(%) | 加速比 |
| ----- | --------------- | ----------------- | ------ | ------ |
| 1节点 | 15895.608               | 11072.04                 | 0.796195      | -      |
| 4节点 | 7941.32               | 22162.16                 | 0.795699      | -      |
| 8节点 | 9915.49               | 22187.04                 | 0.795624      | -      |