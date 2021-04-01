# AI-Rank 软件能力评测 - 训练方向

## 测试方法

区别于其他已有的评测体系，AI-Rank同样重视不同深度学习框架对AI训练任务的支持能力，并设置了多维度指标对框架能力进行综合评测。本任务除设置常规的性能测试指标外，还对不同框架在各个领域所使用的深度学习任务支持能力进行评价。
因此在本任务中，AI-Rank指定了多个热门领域中的10+个有影响力的模型，并从以下几个维度对框架能力进行综合评价：
- 框架性能：在指定硬件型号，模型和数据集的条件下，不同框架训练过程损失值收敛到指定数值（或准确率达到指定数值）的耗时及其他性能指标。
- 任务覆盖率：该框架能够完成的模型任务（即使用被测框架训练该模型能够达到指定的准确率）占总模型任务数的比例。
- 操作系统支持：除Linux类开发环境外，在研究领域和某些行业中还有大量基于Windows的开发和生产环境，能够支持的操作系统系统越丰富，也说明框架有更好的应用能力。

本任务最后会根据以上几方面指标对被测框架产出一个综合评分，用以说明框架的整体能力。

## 硬件环境
基于目前对产业界，科研领域等的调研，AI-Rank指定了2个操作系统+硬件的环境来对框架能力进行测试，后续也会随着业界发展动态更新或添加新的环境：

- Linux操作系统(建议：Ubuntu 16+ 或 CentOS 6+)：
    - GPU：NVIDIA V100-SXM2-16GB
    - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
    - Memory：64 GB
- Windows10：
    - GPU：NVIDIA GeForce RTX 2080Ti
    - CPU：Intel Core i9-9900
    - Memory：32GB

## 模型、数据集和约束条件
AI-Rank从深度学习应用最广泛的几大领域中分别选取了若干有影响力的模型，参与方可以根据自己使用的框架按照标准模型结构进行实现：

| 应用领域 | 模型名称 | 数据集 | 精度约束 |
|----------|---------|---------|---------|
| 图像分类 | ResNet50 | [ImageNet](http://image-net.org/download) | 76.10% classification|
| 目标检测 | Mask R-CNN + FPN | [COCO2017](http://images.cocodataset.org) | box:37.7 mask:33.9 |
| 目标检测 | YOLOv3 | [COCO2017](http://images.cocodataset.org) | 37.0 |
| 图像分割 | DeepLabv3+ | [CityScapes](https://www.cityscapes-dataset.com/) | mIoU: 78.5% |
| 图像分割 | HRNet | [CityScapes](https://www.cityscapes-dataset.com/) | mIoU: 78.0% |
| 语义表示 | BERT | [Wikipedia 2020/01/01](https://dumps.wikimedia.org/enwiki/) | 0.712 Mask-LM accuracy |
| 机器翻译 | Transformer | [WMT](http://data.statmt.org/) | 25.00 BLEU |
| 视频分类 | TSM | kinetics 400 | top1: 0.70 |
| 智能推荐 | DeepFM | -Criteo | AUC: 0.8016 |
| 智能推荐 | Wide&Deep | criteo/censuc-income | AUC：0.80， Loss：0.44 |
| 语义表示 | Word2Vec |  -one billion | ACC：0.530 |

- 为减少软件差异带来的性能影响，最大程度保证公平性，我们对约束条件做如下进一步的解释：
  - 精度约束：在指定验证集上，按照指定的精度评估方法得到的精度，不得低于上述给定的值，例如"76.10% classification"代表分类精度不得低于76.10%;

- 数据集使用规范
    - 必须使用上述指定的数据集进行训练，同时针对不同的评测场景，考虑公平，评审专家组细化规定了数据集中的 训练集、验证集、测试集，并计算了签名，详细参考主页说明。在下面“提交数据”一节中要求提交的代码中能够下载所需数据集并校验签名
        - 训练时，只能使用上述指定数据集中的训练集
        - 精度评测时，只能使用上述指定数据集中的验证集
    - 数据预处理和后处理步骤都要计时

- 模型使用规范
    - 不能使用预训练模型，模型初始化按照常量或随机分布进行
    - 一些操作上可以使用近似计算替代（如：GELU）
    - 不限制模型运行的深度学习框架，如 PaddlePaddle、TensorFlow, ONNX, PyTorch 等均可

- 模型训练规范
    - 可以使用混合精度训练。
    - 精度评估方法和频率。评估方法都按照标准任务的评估方法；除非特殊说明，1个epoch需验证1次精度

## 评价指标

进行软件训练能力评测任务时，考虑到单机能力和分布式能力的体现，参与方应尽量提交包括单机单卡、单机8卡和4机32卡情况下的测试数据。

- 任务覆盖率：该框架能够完成的模型任务（即使用被测框架训练该模型能够达到指定的准确率）占总模型任务数的比例。本指标主要体现框架对业界常用模型的支持全面性。指标根据参与方提交的模型个数确定。
- 操作系统支持数：该框架能够在多少操作系统中完成模型任务。本指标主要体现框架对开发环境多样性的支持能力。指标根据参与方提交的不同系统任务结果确定。
- 模型训练指标，数据通过参与方提交的运行日志获得：
    - 主指标：Time2train，即在特定数据集上训练一个模型使其达到特定精度的用时。单位：sec（秒）。本指标主要关注框架及模型训练的性能，训练开始后计时不暂停。
    - 子指标：
        - 吞吐：单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。本指标主要关注框架及模型训练的性能。
        - 准确率：在eval测试中能够达到的最高准确率。本指标主要关注框架及模型可达到的最好预测精度。
        - 加速比&加速效率：单张显卡Time2train和多张显卡Time2train的比率。本指标主要关注多卡并行训练的性能效果。

## 提交数据

### 提交原则
- 公平公开。所实现的代码能够开源，代码中不能包含作弊的或专为评测设计的代码，也不能包含预先提取评测数据集信息（如：均值、方差等）的代码。
- 训练环境描述清楚。包括：硬件、操作系统、深度学习框架的版本和其它依赖程序的版本。
- 不能使用预训练模型进行初始化。模型参数的初始化使用常量初始化或者某个分布初始化。
- 训练结果可复现。不可复现的训练结果无效。
- 随机性都能被控制。所有的随机性都能通过seed控制，不允许存在不受控的随机性。

### 目录结构

提交目录结构和必要的文件示例如下：

- submitter
    - system
        - system_information.json
        - model1
            - code
                - README.md
            - data
            - log
                - GPUx1.log
                - GPUx8.log
                - GPUx32.log
        - model2
            - code
                - README.md
            - data
            - log
                - GPUx1.log
                - GPUx8.log
                - GPUx32.log

其中，如下目录的名字需要按照实际情况修改：
- submitter目录名为提交公司或组织的名称；
- system 目录名只能为 Linux 或者 Windows；
- modelx目录名只能为评测模型名称或模型名称+深度学习框架名称，每次提交至少包含1个system目录，system目录中可以只提交部分模型；

如下目录名称不用修改：
- code目录中存放评测代码；
- data目录用于存放评测使用的标准数据集；
- log目录用于存放评测日志结果，每次提交时，单机单卡、单机8卡和4机32卡评测日志至少提交一个。

下面详细说明必要文件的内容和格式

### system_information.json 内容要求
描述评测硬件和操作系统信息，如下字段必须存在
|  字段                       | 值（举例）       |
|-----------------------------|-----------------|
| accelerator_memory_capacity |  32 GB          |
| accelerator_name            |  NVIDIA V100-SXM2-32GB      |
| accelerators_per_node       |  8              |
| host_memory_capacity        |  512 GB         |
| host_processor_core_count   |  40             |
| host_processor_name         |  Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz   |
| host_processors_per_node    |  2              |
| host_storage_capacity       |  1 TB           |
| host_storage_type           |  SSD            |
| number_of_nodes             |  1              |
| operating_system            |  Ubuntu 18.04.4 |
| software_stack              |  CUDA 11.0 Update 1, cuDNN 8.0.2  |
| submitter                   |  Baidu          |
| hardware_name               |  某厂商机型      |
| hardware_type               |  datacenter     |

### 每个模型code目录中README.md内容要求
- 数据说明
    - 使用的数据集下载方式说明
    - 数据的预处理或后处理的逻辑说明
    - 训练、精度评测时分别使用的数据范围及遍历顺序说明。即分别使用了数据集中的哪些数据（全部还是部分），以及这些数据使用时按照什么顺序输入

- 模型说明
    - 初始化模型参数的方法，比如常量初始化还是某个分布初始化
    - 模型结构说明，及相比参考模型是否有改变
    - 使用的优化器及其中参数
    - Loss Function说明
    - 精度评估方法和频率说明

- 运行代码的步骤，需给出可执行命令，使其能够执行推理，包括：
    - 在某个特定的硬件环境下，安装所需的相关软件或docker环境等步骤
    - 准备数据、模型、模型优化及相关校验等步骤
    - 执行单机单卡、单机8卡和4机32卡训练的步骤，产生对应的训练日志

### 每个模型log目录中日志格式要求
单机单卡、单机8卡和4机32卡 情况下的日志格式样例如下：
```
- AI-Rank-log 1558631910.424 load_data, checksum:xxxxxxxxxx
- AI-Rank-log 1558631910.424 test_begin
- AI-Rank-log 1558631912.424 eval_accuracy:0.16753999888896942, total_epoch_cnt:1
- AI-Rank-log 1558631913.424 eval_accuracy:0.3207400143146515, total_epoch_cnt:2
- AI-Rank-log 1558631914.424 eval_accuracy:0.4756399989128113, total_epoch_cnt:3
- AI-Rank-log 1558631915.424 eval_accuracy:0.6468799710273743, total_epoch_cnt:4
- AI-Rank-log 1558631916.424 eval_accuracy:0.7605400085449219, total_epoch_cnt:5
- AI-Rank-log 1558631918.454 target_quality_time:8.03sec
- AI-Rank-log 1558631919.424 eval_accuracy:0.8005400085449219, total_epoch_cnt:6
- AI-Rank-log 1558631920.424 eval_accuracy:0.855400085449219, total_epoch_cnt:7
- AI-Rank-log 1558631921.424 eval_accuracy:0.9005400085449219, total_epoch_cnt:8
- AI-Rank-log 1558631922.424 eval_accuracy:0.955400085449219, total_epoch_cnt:9
- AI-Rank-log 1558631922.454 test_finish
- AI-Rank-log 1558631918.454 total_use_time:12.03sec
- AI-Rank-log 1558631918.455 avg_ips:1190images/sec

```
说明：
- 每行以`AI-Rank-log`开始，后接时间戳
- `load_data`：显示所使用数据集的签名
- `test_begin`：测试开始
- `test_finish`：测试结束
- `eval_accuracy`：测试的准确率
- `total_epoch_cnt`：截至当前执行的epoch个数
- `target_quality_time`：`eval_accuracy达到AI-Rank要求的精度时的时间戳` - `test_begin时间戳`
- `total_use_time`：`test_finish时间戳` - `test_begin时间戳`
- `avg_ips`：(`total_epoch_cnt` - `warmup_epoch_cnt`) * `每个epoch的样本总数` / `the_use_time`
    - `warmup_epoch_cnt`，根据自身框架特点，设置的预热epoch数量，预热期间，吞吐速率较低，可不计入avg_ips的计算
    - `the_use_time`，只warmup结束后，到test_finish期间的用时
- 除非特殊说明，训练期间每隔1个epoch进行一次eval测试
- 当eval准确率达到该模型相应要求后，可以停止Time2train指标的计时。这时提交方也可以继续训练，在精度达到自身模型最优值后，停止训练

### summary_metrics.json 结果汇总文件
此文件不在提交范围中。评测日志满足如上要求后，评审专家组会使用工具从日志中生成本次结果汇总文件，放在system目录下，格式为：
|  模型  |  训练卡数   | Time2train(sec)  | 吞吐(samples/sec) | 准确率(%) | 加速比 |
|------- |------------|------------|------------|------------|-----------|
|    -   |    1卡     | -          |    -     |    -     |    -     |
|    -   |    8卡     | -          |    -     |    -     |    -     |
|    -   |    32卡    | -          |    -     |    -     |    -     |
