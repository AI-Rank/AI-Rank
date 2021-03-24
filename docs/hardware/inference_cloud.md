# AI-Rank 硬件能力评测 - 云端推理方向
## 测试方法
针对云端推理的硬件性能测试，AI-Rank采用固定模型、固定参数、固定测试样本集的方法，在此前提下综合评估被测云端设备的性能、能耗等指标。模型和数据集都采用业界有影响力的，可公开获得的版本，以确保测试的权威性和公平性。

## 模型、数据集和约束条件
- 对于云端推理场景，选取了不同应用领域下使用最为频繁的6个模型，并给出了测试集、精度约束、延迟约束和参考模型下载链接：

|应用领域|模型名称|数据集|精度约束|延迟约束|参考模型下载链接|
|-|-|-|-|-|-|
|图像分类|Resnet50|ImageNet（224x224）|>= 99% of FP32 （[Top-1: 76.5%](https://github.com/PaddlePaddle/PaddleClas)）|15ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/ResNet50.tar.gz) [TensorFlow](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)|
|目标检测|Mask R-CNN/ResNet50-FPN|COCO（1200x1200）|>= 99% of FP32（[Box AP: 37.9 Mask AP: 34.2](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.5/docs/MODEL_ZOO_cn.md)）|100ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/mask_rcnn_r50_fpn_1x.tar.gz) [detectron(caffe2)](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md) [detectron2](https://github.com/facebookresearch/detectron2)|
|目标检测|YOLOv3-DarkNet53（608x608）|COCO|>= 99% of FP32 （[Box AP: 38.9](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.5/docs/MODEL_ZOO_cn.md)）|100ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/yolov3_darknet.tar.gz) [gluoncv(mxnet)](https://cv.gluon.ai/model_zoo/detection.html#yolo-v3) |
|图像分割|HRNet_w48|CityScapes|>= 99% of FP32 （[0.8050 mIoU on val, multi-scale_test=false](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/fcn)）| 500ms |[Paddle](https://bj.bcebos.com/paddleseg/airank/hrnetw48_paddle_export.tar) [Pytorch](https://bj.bcebos.com/paddleseg/airank/hrnet48_cityscape.pt)|
|语义表示|BERT |SQUAD 1.1|>= 99% of FP32 （[F1_score: 90.874%](https://github.com/google-research/bert)）|130ms|[Paddle](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/pretrain_language_models/BERT) [TensorFlow](https://github.com/google-research/bert)|
|机器翻译|Transformer（base model）| newstest2014 |>= 99% of FP32 （[BLEU: 26.35](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/machine_translation/transformer)） |130ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/transformer_base.tar.gz)|

- 为减少软件差异带来的性能影响，最大程度保证公平性，我们对约束条件做如下进一步的解释：
  -  精度约束：在指定测试集上，按照指定的精度评估方法得到的精度，不得低于上述给定的值，例如">= 99% of FP32 (76.46%)"代表不得低于99%*76.46%=75.6954%，要求按照第五个有效位进行四舍五入，即精度不得低于75.700%;
  -  延迟约束：在线吞吐测试过程中，要求在约束的时间内处理完所有请求。

- 数据集使用规范
    - 精度评测时，只能使用上述指定数据集中的验证集(validation)
    - 性能评测时，只能使用上述指定数据集中的验证集(validation)
    - 模型量化校准时，只能使用上述指定数据集中的训练集(train))中一部分
    - 这些数据由评审专家组确定，要求固定并计算签名，在下面“提交数据”一节中要求能够下载并校验签名
    - 如下数据预处理步骤可以不计时，但需要在下面“提交数据”一节的README.md中描述清楚。其它步骤需要计时
        - resize和reshape数据
        - 按照某个确定的长度截断文本或填充文本

- 模型使用规范
    - 必须使用与参考模型(fp32)等价的模型，初始的模型参数必须和参考模型一致
    - 可以对初始的模型参数进行量化训练和校准，但所使用的数据满足数据集使用规范，且方法必须可复现优化后的模型。不允许无理由的大量替换模型参数。
    - 一些操作上可以使用近似计算替代（如：GELU）
    - 不限制模型运行的深度学习框架，如 PaddlePaddle、TensorFlow, ONNX, PyTorch 等均可
    - 不允许对非0参数进行裁剪

- 各模型中使用的**单个样本**说明：
   |  模型                   | 单个样本说明 |
   |-------------------------|-------------|
   | Resnet50                |   1张图片   |
   | Mask R-CNN/ResNet50-FPN |   1张图片   |
   | YOLOv3-DarkNet53        |   1张图片   |
   | HRNet_w48               |   1张图片   |
   | BERT                    |   1条语句   |
   | Transformer             |   1条语句   |

## 评价指标

- 离线吞吐：单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。
    - 测试方法：将验证数据集一次性，全部提供给推理程序，推理程序并发推理。计算其整体吞吐速率。

- 在线吞吐：在延迟时间不高于约束值前提下，单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。
    - 测试方法：部署推理服务器，使用测试机模拟client并发请求推理结果，每个请求发送1个sample。逐步增大并发数量，直到响应延迟达到约束的时间（延迟约束）为止。持续保持该并发量，确保延迟始终不高于延迟约束，否则下调并发量。找到一个稳定最大并发值，使得延迟不高于延迟约束。在该并发量下，每秒完成推理的样本数平均值，即在线吞吐。

- 能耗：AI-Rank暂不支持能耗评估。我们将在未来版本中提供。

## 提交数据

### 提交原则
- 公平公开。所实现的代码能够开源，代码中不能包含作弊的或专为评测设计的代码，也不能包含预先提取评测数据集信息（如：均值、方差等）的代码。
- 评测环境描述清楚。包括：硬件、操作系统、深度学习框架的版本和其它依赖程序的版本
- 评测结果可复现。不可复现的评测结果无效。
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
                - accuracy_check.log
                - offline_ips.log
                - online_ips.log
        - model2
            - code
                - README.md
            - data
            - log
                - accuracy_check.log
                - offline_ips.log
                - online_ips.log
        - summary_metrics.json

其中，submitter目录名为提交公司或组织的名称，system 目录名只能为 Linux 或者 Windows，modelx目录名只能为评测模型名称，code目录中存放评测代码，data目录用于存放评测使用的标准数据集，log目录用于存放评测日志结果。下面详细说明必要文件的内容和格式

### system_information.json 内容要求
描述评测硬件和操作系统信息，如下字段必须存在
|  字段                       | 值（举例）       |
|-----------------------------|-----------------|
| accelerator_memory_capacity |  16 GB          |
| accelerator_name            |  NVIDIA T4      |
| accelerators_per_node       |  4              |
| host_memory_capacity        |  64 GB          |
| host_processor_core_count   |  4              |
| host_processor_name         |  Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz   |
| host_processors_per_node    |  2              |
| host_storage_capacity       |  1 TB           |
| host_storage_type           |  SATA HDD       |
| number_of_nodes             |  1              |
| operating_system            |  Ubuntu 18.04.4 |
| software_stack              |  TensorRT 7.2, CUDA 11.0 Update 1, cuDNN 8.0.2  |
| submitter                   |  Baidu          |
| hardware_name               |  某厂商机型      |
| hardware_type               |  datacenter     |

### 每个模型code目录中README.md内容要求
- 数据说明
    - 使用的数据集下载方式说明
    - 数据的预处理或后处理的逻辑说明
    - 自测精度、各性能评测及量化校准时分别使用的数据范围及遍历顺序说明。即分别使用了数据集中的哪些数据，以及这些数据使用时按照什么顺序输入

- 模型说明
    - 使用的fp32模型下载方式、模型结构和参数说明，及相比参考模型是否有改变
    - 如果评测要使用量化模型，则需说明量化方法，要求提供量化方法的代码及可执行命令，能生成量化模型用于评测

- 使用的其它优化方法说明，包括但不限于：使用的TensorRT插件、数据重排序等

- 运行代码的步骤，需给出可执行命令，使其能够执行推理，包括：
    - 在某个特定的硬件环境下，安装所需的相关软件步骤
    - 准备数据、模型、模型优化及相关校验等步骤
    - 分别执行精度、各性能评测的推理步骤，产生评测日志

### 每个模型log目录中日志格式要求
执行精度、各性能评测的推理后，产生的评测日志及格式要求如下：

#### 精度评测日志：accuracy_check.log
```
- AI-Rank-log 1558631910.424 test_begin
- AI-Rank-log 1558631910.424 sampleid:xxxx, result=true
- AI-Rank-log 1558631910.424 sampleid:xxxx, result=false
...
- AI-Rank-log 1558631910.424 sampleid:xxxx, result=true
- AI-Rank-log 1558631910.424 sampleid:xxxx, result=false
- AI-Rank-log 1558631910.424 total_accuracy:0.9999999
- AI-Rank-log 1558631910.424 test_end
```
说明：
- 每行以`AI-Rank-log`开始，后接时间戳
- `test_begin`：测试开始
- `test_finish`：测试结束
- `sampleid`：某个样本的文件名或id，用于确定输入数据
- `result`：推理结果是否正确，true正确，false错误
- `total_accuracy`：整个推理任务的精度

#### 离线吞吐评测日志：offline_ips.log
```
- AI-Rank-log 1558631910.424 test_begin
- AI-Rank-log 1558631910.424 warmup_begin, warmup_samples:100
- AI-Rank-log 1558631910.424 warmup_finish
- AI-Rank-log 1558631912.424 total_accuracy:0.95943342, total_samples_cnt:2
- AI-Rank-log 1558631913.424 total_accuracy:0.95943342, total_samples_cnt:4
- AI-Rank-log 1558631914.424 total_accuracy:0.95943342, total_samples_cnt:6
- AI-Rank-log 1558631915.424 total_accuracy:0.95943342, total_samples_cnt:8
- AI-Rank-log 1558631916.424 total_accuracy:0.95943342, total_samples_cnt:10
- AI-Rank-log 1558631910.424 avg_ips:1190images/sec
- AI-Rank-log 1558631910.424 test_end
```
说明：
- 每行以`AI-Rank-log`开始，后接时间戳
- `test_begin`：测试开始
- `test_finish`：测试结束
- `warmup_begin`：预热开始
- `warmup_finish`：预热结束
- `warmup_samples`：预热样本数量
- `total_accuracy`：已完成测试的所有样本的精度
- `total_samples_cnt`：已完成测试的样本数量
- `avg_ips`：`total_samples_cnt` / 所有时间

#### 在线吞吐评测日志：online_ips.log
注意：以下只是某一特定并发量的日志格式，提交者提交时，只需提交最大并发量的日志即可。
```
- AI-Rank-log 1558631910.424 test_begin
- AI-Rank-log 1558631910.424 target_qps：2000
- AI-Rank-log 1558631912.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:2
- AI-Rank-log 1558631913.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:4
- AI-Rank-log 1558631914.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:6
- AI-Rank-log 1558631915.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:8
- AI-Rank-log 1558631916.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:10
- AI-Rank-log 1558631910.424 test_end
```
说明：
- 每行以`AI-Rank-log`开始，后接时间戳
- `test_begin`：测试开始
- `test_finish`：测试结束
- `target_qps`：最大并发量，也是ips
- `total_accuracy`：已完成测试的所有样本的精度
- `max_latency`：已完成测试的所有样本的最大延迟
- `total_samples_cnt`：已完成测试的所有样本数量

### summary_metrics.json 内容要求
提供工具，从日志中生成此文件数据，格式如下
|  模型  | 离线吞吐(samples/sec)  | 在线吞吐(samples/sec) |
|--------------|--------------|--------------|
|   Resnet50   |    314       |    192       |
