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
|图像分割|DeepLabv3+/Xception65/bn|CityScapes|>= 99% of FP32 （[0.7930 mIoU on val, Output_stride=16，multi-scale_test=false](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.8.0/docs/model_zoo.md)）| 500ms |[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/deeplabv3p_xception65_cityscapes_inference.tar.gz) [TensorFlow](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)|
|语义表示|BERT-Large, Uncased |SQUAD 1.1|>= 99% of FP32 （[SQUAD 1.1 F1/EM 91.0/84.3](https://github.com/google-research/bert)）|130ms|[Paddle](https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz) [TensorFlow](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)|
|机器翻译|Transformer（base model）| newstest2014 |>= 99% of FP32 （[BLEU: 26.35](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/machine_translation/transformer)） |130ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/transformer_base.tar.gz)|

- 为减少软件差异带来的性能影响，最大程度保证公平性，我们对约束条件做如下进一步的解释：
  -  模型：要求必须使用与参考模型等价的模型，不允许重训；
  -  数据集：必须基于上述指定的数据集进行测试；
  -  精度约束：在指定测试集上，按照指定的精度评估方法得到的精度，不得低于上述给定的值，例如">= 99% of FP32 (76.46%)"代表不得低于99%*76.46%=75.6954%，要求按照第五个有效位进行四舍五入，即精度不得低于75.700%;
  -  延迟约束：在线吞吐测试过程中，要求在约束的时间内处理完所有请求。

## 评价指标

- 离线吞吐：单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。
    - 测试方法：将验证数据集一次性，全部提供给推理程序，推理程序并发推理。计算其整体吞吐速率。

- 在线吞吐：在延迟时间不高于约束值前提下，单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。
    - 测试方法：部署推理服务器，使用测试机模拟client并发请求推理结果，每个请求发送1个sample。逐步增大并发数量，直到响应延迟达到约束的时间（延迟约束）为止。持续保持该并发量，确保延迟始终不高于延迟约束，否则下调并发量。找到一个稳定最大并发值，使得延迟不高于延迟约束。在该并发量下，每秒完成推理的样本数平均值，即在线吞吐。

- 能耗：AI-Rank暂不支持能耗评估。我们将在未来版本中提供。

## 提交数据

### 日志格式要求
不同指标测试日志格式如下：

#### 准确率验证
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

#### 离线吞吐
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

#### 在线吞吐
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
- `total_accuracy`：已完成测试的所有样本的准确率
- `max_latency`：已完成测试的所有样本的最大延迟
- `total_samples_cnt`：已完成测试的所有样本数量


### 数据汇总
|  模型  | 离线吞吐(samples/sec)  | 在线吞吐(samples/sec) |
|--------------|--------------|--------------|
|      -       |      -       |      -       |

### 目录结构

提交目录结构示例如下：

- system name
    - system information
    - model1
        - code
        - data
        - log
            - accuracy_check.log
            - offline_ips.log
            - online_ips.log
        - report
    - model2
        - code
        - data
        - log
            - accuracy_check.log
            - offline_ips.log
            - online_ips.log
        - report
    - summary metrics
    - submitter information
