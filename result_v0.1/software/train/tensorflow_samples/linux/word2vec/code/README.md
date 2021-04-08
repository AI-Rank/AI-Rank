# Tensorflow Word2Vec 性能测试

此处给出基于 `Tensorflow-1.15.0` 实现的 Word2Vec 任务的详细复现流程，包括执行环境、Tensorflow版本、环境搭建、复现脚本、测试结果和测试日志，我们将主要测试在参数服务器模式下的性能，使用CPU集群产出模型训练的性能。

## 目录
- [Tensorflow Word2Vec 性能测试](#tensorflow-word2vec-性能测试)
  - [目录](#目录)
  - [一、环境搭建](#一环境搭建)
  - [二、One-Billion 数据集的准备](#二one-billion-数据集的准备)
    - [1. 原始数据集的下载](#1-原始数据集的下载)
    - [2. 数据集预处理](#2-数据集预处理)
  - [三、测试步骤](#三测试步骤)
    - [1. 单节点吞吐测试](#1-单节点吞吐测试)
    - [2. 多节点吞吐测试](#2-多节点吞吐测试)
  - [四、日志数据](#四日志数据)
  - [五、性能指标](#五性能指标)

## 一、环境搭建

我们使用Tensorflow官方提供的[Docker](https://hub.docker.com/r/tensorflow/tensorflow/)镜像，测试环境如下：

- **镜像版本**: `tensorflow/tensorflow:1.15.0`
- **Tensorflow版本**: `1.15.0`

我们在通用K8S集群上成功搭建了测试环境，针对任务训练节点数量及参数服务器数量的配置，集群将分别启动对应数量的pod，并配置分布式训练所需超参，完成参数服务器模式的训练。

执行以下命令以下载及进入容器：

``` bash
docker pull tensorflow/tensorflow:1.15.0
docker run -it --net=host tensorflow/tensorflow:1.15.0 /bin/bash
```


## 二、One-Billion 数据集的准备

训练集选用[1 Billion Word Language Model Benchmark](http://www.statmt.org/lm-benchmark)数据集，该训练集一共包含30294863个文本。

测试集共包含19558个测试样例，每个测试样例由4个词组合构成，依次记为word_a, word_b, word_c, word_d。组合中，前两个词word_a和word_b之间的关系等于后两个词word_c和word_d之间的关系，例如:

```bash
Beijing China Tokyo Japan
write writes go goes
```
所以word2vec的测试任务实际上是一个常见的词类比任务，我们希望通过公式`emb(word_b) - emb(word_a) + emb(word_c)`计算出的词向量和`emb(word_d)`最相近。最终整个模型的评分用成功预测出`word_d`的数量来衡量ACC。


### 1. 原始数据集的下载

```bash
mkdir data
wget --no-check-certificate http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar xzvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
mv 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ data/
```

您也可以从国内源上下载数据，速度更快更稳定。国内源上备用数据下载命令：

```bash
mkdir data
wget --no-check-certificate https://paddlerec.bj.bcebos.com/word2vec/1-billion-word-language-modeling-benchmark-r13output.tar
tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar
mv 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ data/
```

原始预测数据集下载方式：
```
wget --no-check-certificate https://paddlerec.bj.bcebos.com/word2vec/test_dir.tar
tar -xvf test_dir.tar
mkdir test_data
mv data/test_dir/* ./test_data
```

### 2. 数据集预处理

训练集解压后以training-monolingual.tokenized.shuffled目录为预处理目录，预处理主要包括三步，构建词典、数据采样过滤和数据整理。

第一步根据训练语料生成词典，词典格式: 词<空格>词频，出现次数低于5的词被视为低频词，用'UNK'表示：

```bash
python preprocess.py --build_dict --build_dict_corpus_dir data/training-monolingual.tokenized.shuffled --dict_path data/test_build_dict
```

最终得到的词典大小为354051，部分示例如下：

```bash
the 41229870
to 18255101
of 17417283
a 16502705
and 16152335
in 14832296
s 7154338
that 6961093
...
<UNK> 2036007
...
```

第二步数据采样过滤，训练语料中某一个单词被保留下来的概率为：`p_{keep} = \frac{word_count}{down_sampling * corpus_size}`。其中`word_count`为单词在训练集中出现的次数，`corpus_size`为训练集大小，`down_sampling`为下采样参数。

```bash
python preprocess.py --filter_corpus --dict_path data/test_build_dict --input_corpus_dir data/training-monolingual.tokenized.shuffled --output_corpus_dir data/convert_text8 --min_count 5 --downsample 0.001
```

与此同时，这一步会将训练文本转成id的形式，保存在`data/convert_text8`目录下，单词和id的映射文件名为`词典+"word_to_id"`。

最后一步，数据整理。为了方便之后的训练，我们统一将训练数据放在train_data目录下，测试集放在test_data目录下，词表和id映射文件放在thirdparty目录下。同时，为了在多线程分布式训练中达到数据平衡， 从而更好的发挥分布式加速性能，训练集文件个数需尽可能是trainer节点个数和线程数的公倍数，本示例中我们将训练数据重新均匀拆分成1024个文件，您可根据自身情况选择合适的文件个数。

```bash
mkdir thirdparty
mv data/test_build_dict thirdparty/
mv data/test_build_dict_word_to_id_ thirdparty/

python preprocess.py --data_resplit --input_corpus_dir=data/convert_text8 --output_corpus_dir=train_data

mv data/test_dir test_data/
rm -rf data/
```

我们已将上述数据预处理步骤提前完成，并将数据上传至百度云BOS，您可以使用下述代码，下载可以直接参与训练的数据：

``` bash
# 原始数据集已提前上传至百度云BOS，国内网络可快速下载
echo "Begin DownLoad One Billion Data"
wget --no-check-certificate https://paddlerec.bj.bcebos.com/benchmark/word2vec_benchmark_data.tar.gz
echo "Begin Unzip One Billion Data"
tar -xf word2vec_benchmark_data.tar.gz
echo "Get One Billion Data Success"
```

## 三、测试步骤

Word2Vec模型适用的场景是搜索、广告以及推荐，该领域模型训练数据量大，稀疏特征维度巨大，同时模型结构简单，通常使用参数服务器模式进行训练。我们在此主要测试模型单节点性能，以及扩展后的多节点的性能，并统计加速比，使用AUC衡量模型训练效果。

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
    python -u word2vec_distribute.py --sync_mode=False  &> ./log/${mode}/pserver.$i.log &
done

for((i=0;i<${trainer_nums};i++))
do
    export TRAINING_ROLE=TRAINER
    export PADDLE_TRAINER_ID=$i
    python -u word2vec_distribute.py --sync_mode=False &> ./log/${mode}/worker.$i.log &
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
    python -u word2vec_distribute.py --sync_mode=False  &> ./log/${mode}/pserver.$i.log &
done

for((i=0;i<${trainer_nums};i++))
do
    export TRAINING_ROLE=TRAINER
    export PADDLE_TRAINER_ID=$i
    python -u word2vec_distribute.py --sync_mode=False &> ./log/${mode}/worker.$i.log &
done

```

相比单节点，会扩展`PADDLE_PSERVERS_IP_PORT_LIST`以及`PADDLE_WORKERS_IP_PORT_LIST`，同时更改`trainer_nums` 以及 `pserver_nums`配置。

在集群上运行时，按照以下步骤进行分布式训练：
1. 需要首先确定`PADDLE_PSERVERS_IP_PORT_LIST`以及`PADDLE_WORKERS_IP_PORT_LIST`，确保节点之间通过该IP:PORT可访问
2. 针对扮演参数服务器角色的节点，配置` TRAINING_ROLE=PSERVER`，同时给定其顺位`PADDLE_TRAINER_ID`
3. 针对扮演训练器角色的节点，配置`TRAINING_ROLE=TRAINER`，同时给定其顺位`PADDLE_TRAINER_ID`


## 四、日志数据

- [单节点吞吐日志]()

通过以上日志分析，Tensorflow ParameterServer模式， 单节点的吞吐为 **xxx.yyy** `samples/sec`

- [四节点吞吐日志]()

通过以上日志分析，Tensorflow ParameterServer模式， 四节点时，单节点的吞吐为 **xxx.yyy** `samples/sec`， 总吞吐可计算为

- [八节点吞吐日志]()

通过以上日志分析，Tensorflow ParameterServer模式， 八节点的吞吐为 **xxx.yyy** `samples/sec`

## 五、性能指标

|       | Time2train(sec) | 吞吐(samples/sec) | ACC(%) | 加速比 |
| ----- | --------------- | ----------------- | ------ | ------ |
| 1节点 | -               | -                 | -      | -      |
| 4节点 | -               | -                 | -      | -      |
| 8节点 | -               | -                 | -      | -      |