# AI-Rank 硬件能力评测 - 训练方向

## 测试方法

针对硬件在训练方向上的性能测试，AI-Rank采用指定框架和模型的形式，统计在相同数据集上训练过程损失值收敛到指定数值（或准确率达到指定数值）的耗时及其他指标来综合评价硬件的性能。框架，模型和数据集都采用业界有影响力的，可公开获得的版本，以确保测试的权威性和公平性。

## 框架
为减少不同框架实现差异带来的性能影响，保证硬件性能数据的可对比性，应尽量选取相同的框架进行测试，但考虑到不同硬件和框架间的适配性问题，本任务中可以选用如下3个框架中的一个来提交测试结果：
- [PaddlePaddle 2.0.0](https://www.paddlepaddle.org.cn/install/quick/zh/2.0rc-linux-pip)
- [TensorFlow 2.4.0](https://tensorflow.google.cn/install/)
- [PyTorch 1.7.0](https://pytorch.org/get-started/locally/#linux-installation)

注意：在测试结果公布时，不同硬件测试结果间，只有使用了相同框架和模型的测试数据才具有比较意义。

## 模型和数据集
AI-Rank 选择最具代表性的4个模型，作为衡量硬件计算能力的测试“程序”：
| 应用领域 | 模型名称 | 数据集 | 精度约束 | 参考模型实现 |
|----------|---------|---------|---------|---------|
| 图像分类 | ResNet50 | ImageNet | 75.90% classification | [PaddlePaddle](https://github.com/PaddlePaddle/PaddleClas/tree/master/ppcls/modeling/architectures) [TensorFlow](https://github.com/tensorflow/models/tree/master/official/vision/image_classification) [PyTorch](https://github.com/pytorch/vision/tree/master/references/classification) |
| 目标检测 | Mask R-CNN + FPN | COCO2017 | box:37.7 mask:33.9 | [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection/tree/master/ppdet/modeling/architectures) [TensorFlow](https://github.com/tensorflow/models/tree/master/official/vision/detection/modeling) [PyTorch](https://github.com/open-mmlab/mmdetection) |
| 语义表示 | BERT | Wikipedia 2020/01/01 | 0.712 Mask-LM accuracy | [PaddlePaddle](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT) [TensorFlow](https://github.com/tensorflow/models/tree/master/official/nlp/bert) [PyTorch](https://github.com/huggingface/transformers) |
| 机器翻译 | Transformer | WMT | big model, bleu: 25.00 | [PaddlePaddle](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/transformer) [TensorFlow](https://github.com/tensorflow/tensor2tensor) [PyTorch](https://github.com/pytorch/fairseq) |

为进一步减少软件差异带来的性能影响，最大程度保证公平性，我们对约束条件做如下进一步的解释：
  - 精度约束：在指定验证集上，按照指定的精度评估方法得到的精度，不得低于上述给定的值，例如"75.90% classification"代表分类精度不得低于75.90%;
  - 参考模型实现：在本任务进行评测时，上述模型的实现只能从上面公开版本中获取

## 评价指标

进行硬件训练性能评测任务时，原则上不对硬件的数量和规模进行限制，但考虑到产业界实际应用情况，参与方应尽量提交包括单机单卡、单机8卡和4机32卡情况下的性能测试数据。除此之外，AI-Rank也更鼓励参与方提交多样化的硬件测试结果，不仅仅追求单一速度指标，也展示更多具有性价比的硬件性能，为更多企业的方案选择提供实际参考数据。

- 模型训练指标，数据通过参与方提交的运行日志获得：
    - 主指标：Time2train，即在特定数据集上训练一个模型使其达到特定精度的用时。单位：sec（秒）。本指标主要关注被测硬件的计算性能，训练开始后计时不暂停。
    - 子指标：
        - 吞吐：单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。本指标主要关注被测硬件的性能。
        - 加速比&加速效率：单张显卡Time2train和多张显卡Time2train的比率。本指标主要关注多卡并行训练的性能效果，尤其是硬件建通信机制的性能。
        - 能耗：AI-Rank暂不支持能耗评估。我们将在未来版本中提供。

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
- system 目录名按照具体使用的系统命名，例如 Linux、Windows等；
- modelx目录名只能为评测模型名称或模型名称+深度学习框架名称，每次提交至少包含1个system目录，system目录中可以只提交部分模型；

如下目录名称不用修改：
- code目录中存放评测代码；
- data目录用于存放评测使用的标准数据集；
- log目录用于存放评测日志结果，每次提交时，单机单卡、单机8卡和4机32卡评测日志至少提交一个。

下面详细说明必要文件的内容和格式

### system_information.json 内容要求
描述评测硬件和操作系统信息，如下字段必须存在
|  字段                       | 值（举例）       |
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
- AI-Rank-log 1558631922.454 test_finish
- AI-Rank-log 1558631918.454 target_quality_time:8.03sec
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
- `avg_ips`：(`total_epoch_cnt` - `warmup_epoch_cnt`) * `每个epoch的样本总数` / `the_use_time`
    - `warmup_epoch_cnt`，根据自身框架特点，设置的预热epoch数量，预热期间，吞吐速率较低，可不计入avg_ips的计算
    - `the_use_time`，只warmup结束后，到test_finish期间的用时
- 除非特殊说明，训练期间每隔1个epoch进行一次eval测试
- 当eval准确率达到该模型相应要求后，可以停止Time2train指标的计时，训练即可停止

### summary_metrics.json 结果汇总文件
此文件不在提交范围中。评测日志满足如上要求后，评审专家组会使用工具从日志中生成本次结果汇总文件，放在system目录下，格式为：
|  模型  |  训练卡数   | Time2train(sec)  | 吞吐(samples/sec) | 准确率(%) | 加速比 |
|------- |------------|------------|------------|------------|-----------|
|    -   |    1卡     | -          |    -     |    -     |    -     |
|    -   |    8卡     | -          |    -     |    -     |    -     |
|    -   |    32卡    | -          |    -     |    -     |    -     |
