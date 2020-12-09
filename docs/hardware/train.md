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
|模型名称 | 应用领域 | 训练数据集 |
|----------|---------|---------|
|ResNet50 | 图像分类 | ImageNet |
|Mask R-CNN + FPN | 目标检测 | COCO2017 |
|BERT | 语义表示 | Wikipedia 2020/01/01 |
|Transformer | 机器翻译 | WMT |

为进一步减少软件差异带来的性能影响，最大程度保证公平性，在本任务进行测试时，上述模型的实现只能从如下公开版本中获取：

|模型名称 | PaddlePaddle | TensorFlow | PyTorch |
|----------|---------|---------|---------|
|ResNet50 | https://github.com/PaddlePaddle/PaddleClas/tree/master/ppcls/modeling/architectures | https://github.com/tensorflow/models/tree/master/official/vision/image_classification | https://github.com/pytorch/vision/tree/master/references/classification |
|Mask R-CNN + FPN | https://github.com/PaddlePaddle/PaddleDetection/tree/master/ppdet/modeling/architectures | https://github.com/tensorflow/models/tree/master/official/vision/detection/modeling |https://github.com/open-mmlab/mmdetection|
|BERT | https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT | https://github.com/tensorflow/models/tree/master/official/nlp/bert | https://github.com/huggingface/transformers|
|Transformer | https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/machine_translation/transformer | https://github.com/tensorflow/tensor2tensor | https://github.com/pytorch/fairseq|

## 评价指标

进行硬件训练性能评测任务时，原则上不对硬件的数量和规模进行限制，但考虑到产业界实际应用情况，参与方应尽量提交包括单机单卡，单机8卡，4机32卡情况下的性能测试数据。除此之外，AI-Rank也更鼓励参与方提交多样化的硬件测试结果，不仅仅追求单一速度指标，也展示更多具有性价比的硬件性能，为更多企业的方案选择提供实际参考数据。

- 主指标：Time2Train：在特定数据集上训练一个模型使其达到特定精度的用时。单位：sec（秒）。本指标主要关注被测硬件的计算性能，指标根据参与者提交的运行日志确定。

特定精度的定义如下：
|模型名称 | 目标精度|
|--------------|------------|
|ResNet50 | 75.90% classification|
|Mask R-CNN + FPN | -|
|BERT | -|
|Transformer | -|

- 子指标：
    - 吞吐：单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。本指标主要关注被测硬件的性能。指标根据参与者提交的运行日志确定。
    - 加速比&加速效率：单张显卡Time2train和多张显卡Time2train的比率。本指标主要关注多卡并行训练的性能效果，尤其是硬件建通信机制的性能。指标根据参与者提交的运行日志换算获得。
    - 能耗：AI-Rank暂不支持能耗评估。我们将在未来版本中提供。

## 提交数据
提交的材料应包括：
- 参与的任务：选用的框架、模型等。
- 被测系统信息：硬件型号，相关系统软件版本信息，有条件的参与方可以提交测试时使用的docker环境。
- 源码：测试所需源码，如算子、执行脚本等。
- 数据：测试所需的数据，包括数据下载方法、数据预处理方法等。
- 日志：提交方自行测试的日志。
- 报告及结果：可用于评审组复现的《测试&复现报告》。
- 数据汇总：将核心指标按下文中格式要求汇总整理。
- 参与方信息：单位名称等。

### 日志格式要求
日志格式样例如下：
```
- AI-Rank-log 1558631910.424 test_begin
- AI-Rank-log 1558631912.424 eval_accuracy:0.16753999888896942, total_epoch_cnt:2
- AI-Rank-log 1558631913.424 eval_accuracy:0.3207400143146515, total_epoch_cnt:4
- AI-Rank-log 1558631914.424 eval_accuracy:0.4756399989128113, total_epoch_cnt:6
- AI-Rank-log 1558631915.424 eval_accuracy:0.6468799710273743, total_epoch_cnt:8
- AI-Rank-log 1558631916.424 eval_accuracy:0.7605400085449219, total_epoch_cnt:10
- AI-Rank-log 1558631922.454 test_finish
- AI-Rank-log 1558631918.454 target_quality_time:8.03sec
- AI-Rank-log 1558631918.455 avg_ips:1190images/sec

```
说明：
- 每行以`AI-Rank-log`开始，后接时间戳
- `test_begin`：测试开始
- `test_finish`：测试结束
- `eval_accuracy`：测试的准确率
- `total_epoch_cnt`：截至当前执行的epoch个数
- `target_quality_time`：`eval_accuracy达到AI-Rank要求的精度时的时间戳` - `test_begin时间戳`
- `avg_ips`：(`total_epoch_cnt` - `warmup_epoch_cnt`) * `每个epoch的样本总数` / `the_use_time`
    - `warmup_epoch_cnt`，根据自身框架特点，设置的预热epoch数量，预热期间，吞吐速率较低，可不计入avg_ips的计算
    - `the_use_time`，只warmup结束后，到test_finish期间的用时
- 训练期间每隔N个epoch进行一次eval测试，N由提交方自定义
- 当eval准确率达到该模型相应要求后，测试即可停止

### 数据汇总
数据汇总格式如下：
- ResNet50

|              | Time2train(sec)  | 吞吐(samples/sec) | 加速比 | 能耗(暂不需要填写) |
|--------------|------------|------------|------------|-----------|
| 1卡          | -          |    -     |    -     |    -     |
| 8卡          | -          |    -     |    -     |    -     |
| 32卡          | -          |    -     |    -     |    -     |

### 目录结构

提交目录结构示例如下：

- system name
    - system information
    - ResNet50
        - code
        - data
        - log
            - GPUx1.log
            - GPUx8.log
            - GPUx32.log
        - report
    - Mask R-CNN + FPN
        - code
        - data
        - log
            - GPUx1.log
            - GPUx8.log
            - GPUx32.log
        - report
    - summary metrics
    - submitter information
