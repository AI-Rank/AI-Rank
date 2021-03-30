# AI-Rank 软件能力评测 - 终端推理方向

## 测试方法
AI-Rank针对终端推理框架设置了几个维度的不同指标来评价其综合能力，除常规的性能测试指标外，还考察推理框架对市面上各种常见终端硬件的适配能力，因此本任务包括以下几个维度：
- 框架性能：在固定模型、固定参数及固定测试样本集的条件下，测试不同框架的各项性能指标。模型和数据集都采用业界有影响力的，可公开获得的版本，以确保测试的权威性和公平性。
- 硬件适配能力：选取当前市面上常见的终端硬件，分为必选硬件和可选硬件两类，被测框架需至少完成所有必选硬件的测试任务，除必选硬件外，框架能够在可选硬件上完成的任务数量越多，说明其适配能力越好。
- 能耗：对于终端硬件，能耗是一项重要指标，但目前尚未对该指标设置具体评测方案，会在后续版本中进行设置。

## 硬件环境
基于目前对产业界，科研领域等的调研，AI-Rank指定了如下终端硬件，并不限定操作系统、驱动版本，后续也会随着业界发展动态更新或添加新的环境：
- 必选硬件：
    - 高通865（小米10、vivo iQOO Neo3 5G、OnePlus 8T 5G等）
    - 高通835（小米MIX2等）
    - 高通625（红米6pro等）
    - 麒麟9905G（P40pro等）
    - RK3399（Firefly RK3399等）
    - 树莓派4B
- 可选硬件：
    - RK3288
    - iPhone 11
    - iPhone XR
    - iPhone6s
    - iPhone5s
    - Jetson AGX Xavier
    - Jetson Nano
    - Jetson TX2

## 模型、数据集和约束条件
- 对于移动端推理场景，选取了不同应用领域下使用最为频繁的8个模型，并给出了测试集、精度约束、延迟约束和参考模型下载链接：

|应用领域|模型名称|数据集|精度约束|延迟约束|参考模型下载链接|
|-|-|-|-|-|-|
|图像分类|MobileNetV1|ImageNet (224x224）|>= 99% of FP32（[Top-1: 70.99%](https://github.com/PaddlePaddle/PaddleClas)）|50ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/MobileNetV1.tar.gz) [TensorFlow](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)|
|图像分类|MobileNetV2|ImageNet (224x224）|>= 99% of FP32（[Top-1: 72.15%](https://github.com/PaddlePaddle/PaddleClas)）|50ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/MobileNetV2.tar.gz) [TensorFlow](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz)|
|图像分类|MobileNetV3_large_x1_0|ImageNet (224x224）|>= 99% of FP32（[Top-1: 75.32%](https://github.com/PaddlePaddle/PaddleClas)）|50ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/MobileNetV3_large_x1_0.tar.gz) [TensorFlow](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz)|
|图像分类|MobileNetV3_small_x1_0|ImageNet (224x224）|>= 99% of FP32（[Top-1: 68.24%](https://github.com/PaddlePaddle/PaddleClas)）|50ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/MobileNetV3_small_x1_0.tar.gz) [TensorFlow](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_1.0_float.tgz)|
|图像分类|Resnet50|ImageNet (224x224）|>= 99% of FP32（[Top-1: 76.5%](https://github.com/PaddlePaddle/PaddleClas)）|50ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/ResNet50.tar.gz) [TensorFlow](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)|
|目标检测|SSD-MobileNetV3_large（320x320）|COCO|>= 99% of FP32（[COCO mAP: 22.6](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.5/docs/MODEL_ZOO_cn.md)）|100ms|[Paddle](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/ssdlite_mobilenet_v3_large.tar) [TensorFlow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz)|
|目标检测|Yolov3-MobileNetV1（608x608）|COCO|>= 99% of FP32（[Box AP: 29.3](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.5/docs/MODEL_ZOO_cn.md)）|100ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/yolov3_mobilenet_v1.tar.gz) [gluoncv(mxnet)](https://cv.gluon.ai/model_zoo/detection.html#yolo-v3)|
|图像分割|HRNet_w18|CityScapes|>= 99% of FP32 （[0.7850 mIoU on val, multi-scale_test=false](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/fcn)）|1000ms|[Paddle](https://bj.bcebos.com/paddleseg/airank/hrnetw18_paddle_export.tar) [Pytorch](https://bj.bcebos.com/paddleseg/airank/hrnet18_cityscape.pt)|

- 为减少软件差异带来的性能影响，最大程度保证公平性，我们对约束条件做如下进一步的解释：
  -  模型：要求必须使用与参考模型等价的模型，参与方可以根据自己使用的框架按照标准模型结构进行实现，可基于提供的校准数据进行后量化，但不允许重训；
  -  数据集：必须基于上述指定的数据集进行测试；
  -  精度约束：在指定测试集上，按照指定的精度评估方法得到的精度，不得低于上述给定的值，例如">= 99% of FP32 (76.46%)"代表不得低于99%*76.46%=75.6954%，要求按照第五个有效位进行四舍五入，即精度不得低于75.700%;
  -  延迟约束：最大并发推理量测试过程中，要求在约束的时间内处理完所有请求。

## 评价指标
- 硬件适配能力：该框架能够在给定硬件上（包括可选和必选）完成的任务数占总任务数的比例。本指标主要体现框架对业界常用硬件的适配全面性。指标根据参与者提交的任务个数确定。
- 性能指标：
    - 时延：N次推理，每次1个sample，计算每次推理的用时。时延取N次用时的90分位值。单位：ms（毫秒）
        - 测试方法：取1000个样本，每次推理一个样本，计算该样本推理延迟时间。1000个延迟时间升序排序，取90分位值。
    - 离线吞吐：单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。
        - 测试方法：将验证数据集一次性，全部提供给推理程序，推理程序并发推理。计算其整体吞吐速率。
    - 最大并发推理量：在延迟时间不高于约束值前提下，最大支持的一个批次的BatchSize值。
        - 测试方法：部署推理程序到终端，编写测试程序，加载验证数据集，每次加载N个samples。N的数量逐步增大，直到响应延迟达到约束的时间（延迟约束）为止。持续保持N的值，确保延迟始终不高于延迟约束，否则下调N值。找到一个稳定N值，使得延迟不高于延迟约束。N值即最大并发推理量。

- 内存占用：推理期间，推理模型最大使用内存量。单位：MB。
    - 测试方法：在`最大并发推理量`测试过程中，最大的内存占用值。
- 能耗：本次暂不设置该指标，会在后续版本中进行设置。

# 提交数据

## 日志格式要求
不同指标测试日志格式如下：

### 准确率验证
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
- `sampleid`：某个sample的文件名
- `result`：推理结果是否正确，true正确，false错误
- `total_accuracy`：整个推理的准确率

### 延迟
```
- AI-Rank-log 1558631910.424 test_begin
- AI-Rank-log 1558631910.424 latency_case1_begin
- AI-Rank-log 1558631910.424 latency_case1_finish
- AI-Rank-log 1558631910.424 latency_case2_begin
- AI-Rank-log 1558631910.424 latency_case2_finish
- AI-Rank-log 1558631910.424 latency_case3_begin
- AI-Rank-log 1558631910.424 latency_case3_finish
...
- AI-Rank-log 1558631910.424 latency_case1000_begin
- AI-Rank-log 1558631910.424 latency_case1000_finish
- AI-Rank-log 1558631910.424 latency_case1000_finish
- AI-Rank-log 1558631910.424 90th_percentile_latency:20ms
- AI-Rank-log 1558631910.424 test_end
```
说明：
- 每行以`AI-Rank-log`开始，后接时间戳
- `test_begin`：测试开始
- `test_finish`：测试结束
- `latency_caseN_begin`：第N次测试开始
- `latency_caseN_finish`：第N次测试结束
- `90th_percentile_latency`：取90分位的延迟时间


### 离线吞吐
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
- `warmup_samples`：预热samples数量
- `total_accuracy`：以完成训练的sample，准确率
- `total_samples_cnt`：已完成的sample数量
- `avg_ips`：`total_samples_cnt` / 所有时间

### 最大并发推理量&内存占用
注意：以下只是某一特定并发量的日志格式，提交者提交时，只需提交最大并发量的日志即可。
```
- AI-Rank-log 1558631910.424 test_begin
- AI-Rank-log 1558631910.424 samples_cnt_each_case:100
- AI-Rank-log 1558631912.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:2, max_memory_use:10M
- AI-Rank-log 1558631913.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:4, max_memory_use:10M
- AI-Rank-log 1558631914.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:6, max_memory_use:10M
- AI-Rank-log 1558631915.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:8, max_memory_use:10M
- AI-Rank-log 1558631916.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:10, max_memory_use:10M
- AI-Rank-log 1558631910.424 test_end
```
说明：
- 每行以`AI-Rank-log`开始，后接时间戳
- `test_begin`：测试开始
- `test_finish`：测试结束
- `samples_cnt_each_case`：每次发起预测的samples数量
- `total_accuracy`：已完成测试的所有样本的准确率
- `max_latency`：已完成测试的所有样本的最大延迟
- `total_samples_cnt`：已完成测试的所有样本数量
- `max_memory_use`：自测试以来的最大内存占用值

## 数据汇总
|  模型  | 硬件 | 时延（ms） | 离线吞吐(samples/sec) | 最大并发推理量(samples/sec) | 内存占用(MB) |
|--------------|--------------|--------------|--------------|--------------|--------------|
|      -       |      -       |      -       |      -       |      -       |      -       |

## 目录结构

提交目录结构示例如下：

- system name
    - system information
    - model1-qualcomm865
        - code
        - data
        - log
            - accuracy_check.log
            - latency.log
            - offline_ips.log
            - max_qps_max_memory_use.log
        - report
    - model2-RK3399
        - code
        - data
        - log
            - accuracy_check.log
            - latency.log
            - offline_ips.log
            - max_qps_max_memory_use.log
        - report
    - summary metrics
    - submitter information
