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

- Linux操作系统：
    - GPU：NVIDIA V100-SXM2-16GB
    - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
    - Memory：64 GB
- Windows10：
    - GPU：NVIDIA GeForce RTX 2080Ti
    - CPU：Intel Core i9-9900
    - Memory：32GB

## 模型任务
AI-Rank从深度学习应用最广泛的几大领域中分别选取了若干有影响力的模型，参与方可以根据自己使用的框架按照标准模型结构进行实现：
|模型名称 | 应用领域 | 训练数据集 | 入选理由 |
|----------|---------|---------|---------|
|ResNet50 | 图像分类 | ImageNet | 最常用的图片分类模型 |
|Mask R-CNN + FPN | 目标检测 | - | -|
|YOLOv3 | 目标检测 | - | -|
|DeepLabv3+ | 图像分割 | - | -|
|HRNet | 图像分割 | - | -|
|BERT | 语义表示 | - | -|
|Transformer | 机器翻译 | - | -|
|CycleGAN | 图像生成 | - | -|
|pix2pix | 图像生成 | - | -|
|TSM | 视频分类 | - | -|
|DSSM | 智能推荐 | - | -|
|DeepFM | 智能推荐 | - | -|
|Wide&Deep | 智能推荐 | - | -|
|Word2Vec | 语义表示 | - | -|

## 评价指标

进行软件训练能力评测任务时，考虑到单机能力和分布式能力的体现，参与方应尽量提交包括单机单卡，单机8卡和4机32卡情况下的测试数据。

- 任务覆盖率：该框架能够完成的模型任务（即使用被测框架训练该模型能够达到指定的准确率）占总模型任务数的比例。本指标主要体现框架对业界常用模型的支持全面性。指标根据参与者提交的模型个数确定。
- 操作系统支持数：该框架能够在多少操作系统中完成模型任务。本指标主要体现框架对开发环境多样性的支持能力。指标根据参与者提交的不同系统任务结果确定。
- 性能指标：
    - 主指标：Time2train，即在特定数据集上训练一个模型使其达到特定精度的用时。单位：sec（秒）。本指标主要关注框架及模型训练的性能。指标根据参与者提交的运行日志确定，具体模型精度要求如下：
    
|模型名称 | 目标精度|
|--------------|------------|
|ResNet50 | 75.90% classification|
|Mask R-CNN + FPN | -|
|YOLOv3 | -|
|DeepLabv3+ | -|
|HRNet | -|
|BERT | -|
|Transformer | -|
|CycleGAN | -|
|pix2pix | -|
|TSM | -|
|DSSM | -|
|DeepFM | -|
|Wide&Deep | -|
|Word2Vec | -|

    - 子指标：
        - 吞吐：单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。本指标主要关注框架及模型训练的性能。指标根据参与者提交的运行日志确定。
        - 准确率：在eval测试中能够达到的最高准确率。本指标主要关注框架及模型可达到的最好预测精度。指标根据参与者提交的运行日志确定。
        - 加速比&加速效率：单张显卡Time2train和多张显卡Time2train的比率。本指标主要关注多卡并行训练的性能效果。指标根据参与者提交的运行日志换算获得。

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
- AI-Rank-log 1558631918.454 target_quality_time:8.03sec
- AI-Rank-log 1558631919.424 eval_accuracy:0.8005400085449219, total_epoch_cnt:12
- AI-Rank-log 1558631920.424 eval_accuracy:0.855400085449219, total_epoch_cnt:14
- AI-Rank-log 1558631921.424 eval_accuracy:0.9005400085449219, total_epoch_cnt:16
- AI-Rank-log 1558631922.424 eval_accuracy:0.955400085449219, total_epoch_cnt:18
- AI-Rank-log 1558631922.454 test_finish
- AI-Rank-log 1558631918.454 total_use_time:12.03sec
- AI-Rank-log 1558631918.455 avg_ips:1190images/sec

```
说明：
- 每行以`AI-Rank-log`开始，后接时间戳
- `test_begin`：测试开始
- `test_finish`：测试结束
- `eval_accuracy`：测试的准确率
- `total_epoch_cnt`：截至当前执行的epoch个数
- `target_quality_time`：`eval_accuracy达到AI-Rank要求的精度时的时间戳` - `test_begin时间戳`
- `total_use_time`：`test_finish时间戳` - `test_begin时间戳`
- `avg_ips`：(`total_epoch_cnt` - `warmup_epoch_cnt`) * `每个epoch的样本总数` / `the_use_time`
    - `warmup_epoch_cnt`，根据自身框架特点，设置的预热epoch数量，预热期间，吞吐速率较低，可不计入avg_ips的计算
    - `the_use_time`，只warmup结束后，到test_finish期间的用时
- 训练期间每隔N个epoch进行一次eval测试，N由提交方自定义
- 当eval准确率达到该模型相应要求后，提交方跟进自身经验，在精度达到自身模型最优值后，停止测试

### 数据汇总
数据汇总格式如下：
- 模型1

|              | Time2train(sec)  | 吞吐(samples/sec) | 准确率(%) | 加速比 |
|--------------|------------|------------|------------|-----------|
| 1卡          | -          |    -     |    -     |    -     |
| 32卡          | -          |    -     |    -     |    -     |


### 目录结构

提交目录结构示例如下：

- system name
    - system information
    - model1
        - code
        - data
        - log
            - GPUx1.log
            - GPUx32.log
        - report
    - model2
        - code
        - data
        - log
            - GPUx1.log
            - GPUx32.log
        - report
    - summary metrics
    - submitter information
