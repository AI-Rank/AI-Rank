# AI-Rank 硬件能力评测 - 终端推理方向
## 测试方法
针对终端推理的硬件性能测试，AI-Rank采用固定框架、固定模型、固定训练参数、固定测试样本集的方法，在此前提下综合评估被测云端设备的性能、能耗等指标。框架，模型和数据集都采用业界有影响力的，可公开获得的版本，以确保测试的权威性和公平性。

## 模型、数据集和约束条件
- 对于移动端推理场景，选取了不同应用领域下使用最为频繁的8个模型，并给出了测试集、精度约束、延迟约束和参考模型下载链接：

|应用领域|模型名称|数据集|精度约束|延迟约束|参考模型下载链接|
|-|-|-|-|-|-|
|图像分类|MobileNetV1|ImageNet (224x224）|>= 99% of FP32（[Top-1: 70.99%](https://github.com/PaddlePaddle/PaddleClas)）|200ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/MobileNetV1.tar.gz) [TensorFlow](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)|
|图像分类|MobileNetV2|ImageNet (224x224）|>= 99% of FP32（[Top-1: 72.15%](https://github.com/PaddlePaddle/PaddleClas)）|150ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/MobileNetV2.tar.gz) [TensorFlow](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz)|
|图像分类|MobileNetV3_large_x1_0|ImageNet (224x224）|>= 99% of FP32（[Top-1: 75.32%](https://github.com/PaddlePaddle/PaddleClas)）|150ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/MobileNetV3_large_x1_0.tar.gz) [TensorFlow](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz)|
|图像分类|MobileNetV3_small_x1_0|ImageNet (224x224）|>= 99% of FP32（[Top-1: 68.24%](https://github.com/PaddlePaddle/PaddleClas)）|80ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/MobileNetV3_small_x1_0.tar.gz) [TensorFlow](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_1.0_float.tgz)|
|图像分类|Resnet50|ImageNet (224x224）|>= 99% of FP32（[Top-1: 76.5%](https://github.com/PaddlePaddle/PaddleClas)）|1100ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/ResNet50.tar.gz) [TensorFlow](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)|
|目标检测|SSD-MobileNetV1（300x300）|Pascal VOC|>= 99% of FP32（[Box AP: 73.2](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.5/docs/MODEL_ZOO_cn.md)）|250ms|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/ssd_mobilenet_v1_voc.tar.gz) [Pytorch](https://github.com/chuanqi305/MobileNet-SSD#mobilenet-ssd)|
|目标检测|Yolov3-MobileNetV1（608x608）|COCO|>= 99% of FP32（[Box AP: 29.3](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.5/docs/MODEL_ZOO_cn.md)）|N/A|[Paddle](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/models/Paddle/yolov3_mobilenet_v1.tar.gz) [gluoncv(mxnet)](https://cv.gluon.ai/model_zoo/detection.html#yolo-v3)|
|图像分割|HRNet_w18|CityScapes|>= 99% of FP32 （[0.7850 mIoU on val, multi-scale_test=false](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0/configs/fcn)）|N/A|[Paddle](https://bj.bcebos.com/paddleseg/airank/hrnetw18_paddle_export.tar) [Pytorch](https://bj.bcebos.com/paddleseg/airank/hrnet18_cityscape.pt)|

- 为减少软件差异带来的性能影响，最大程度保证公平性，我们对约束条件做如下进一步的解释：
  -  精度约束：在指定测试集上，按照指定的精度评估方法得到的精度，不得低于上述给定的值，例如">= 99% of FP32 (76.46%)"代表不得低于99%*76.46%=75.6954%，要求按照第五个有效位进行四舍五入，即精度不得低于75.700%;
  -  延迟约束：最大并发推理量测试过程中，要求在约束的时间内处理完所有请求。

- 数据集使用规范
    - 必须使用上述指定的数据集进行评测，同时针对不同的评测场景，考虑公平，评审专家组细化规定了数据集中的 训练集、验证集、测试集，并计算了签名，详细参考主页说明。在下面“提交数据”一节中要求提交的代码中能够下载所需数据集并校验签名
        - 精度评测时，只能使用上述指定数据集中的验证集
        - 性能评测时，只能使用上述指定数据集中的验证集
        - 模型量化校准时，只能使用上述指定数据集中的训练集中一部分，由评审专家组提供
    - 如下数据预处理和后处理步骤可以不计时，但需要在下面“提交数据”一节的README.md中描述清楚预处理和后处理逻辑。其它步骤需要计时
        - resize和reshape数据
        - 按照某个确定的长度截断文本或填充文本
        - 推理结果中的数字转为文本打印

- 模型使用规范
    - 必须使用与参考模型等价的模型，初始的模型参数必须和参考模型一致
    - 可以对初始的模型参数进行量化校准，但所使用的数据满足数据集使用规范，且方法必须可复现优化后的模型，不允许重训。不允许无理由的大量替换模型参数。
    - 一些操作上可以使用近似计算替代（如：GELU）
    - 不限制模型运行的深度学习框架，如 PaddleLite、TensorFlow Lite、NCNN 和 LibTorch 等均可
    - 不允许对非0参数进行裁剪

- 各模型中使用的**单个样本**说明：
   |  模型                   | 单个样本说明 |
   |-------------------------|-------------|
   | MobileNetV1             |   1张图片   |
   | MobileNetV2             |   1张图片   |
   | MobileNetV3_large_x1_0  |   1张图片   |
   | MobileNetV3_small_x1_0  |   1张图片   |
   | Resnet50                |   1张图片   |
   | SSD-MobileNetV3_large   |   1张图片   |
   | Yolov3-MobileNetV1      |   1张图片   |
   | HRNet_w18               |   1张图片   |

## 评价指标
均在单线程环境下运行：
- 时延：N次推理，每次1个sample，计算每次推理的用时。时延取N次用时的90分位值。单位：ms（毫秒）
    - 测试方法：取1000个样本，每次推理一个样本，计算该样本推理延迟时间。1000个延迟时间升序排序，取90分位值。
- 离线吞吐：单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。
    - 测试方法：将验证数据集一次性，全部提供给推理程序，推理程序并发推理。计算其整体吞吐速率。
- 最大并发推理量：在延迟时间不高于约束值前提下，最大支持的一个批次的BatchSize值。
    - 测试方法：部署推理程序到终端，编写测试程序，加载验证数据集，每次加载N个samples。N的数量逐步增大，直到响应延迟达到约束的时间（延迟约束）为止。持续保持N的值，确保延迟始终不高于延迟约束，否则下调N值。找到一个稳定N值，使得延迟不高于延迟约束。N值即最大并发推理量。

- 内存占用：本次暂不设置该指标，会在后续版本中进行设置。

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
                - architecture
                    - accuracy_check.log
                    - latency.log
                    - offline_ips.log
                    - max_qps_max_memory_use.log
        - model2
            - code
                - README.md
            - data
            - log
                - architecture
                    - accuracy_check.log
                    - latency.log
                    - offline_ips.log
                    - max_qps_max_memory_use.log

其中，如下目录的名字需要按照实际情况修改：
- submitter目录名为提交公司或组织的名称；
- system 目录名为硬件名称，如qualcomm865、RK3399等；
- modelx目录名只能为评测模型名称或模型名称+深度学习框架名称，每次提交可以只提交部分模型；
- architecture目录名只能为armv7或armv8等，每次至少提交一个；

如下目录名称不用修改：
- code目录中存放评测代码；
- data目录用于存放评测使用的标准数据集；
- log目录用于存放评测日志结果，每次提交时，在线和离线吞吐评测日志至少提交一个，精度评测日志必须提交。

下面详细说明必要文件的内容和格式

### system_information.json 内容要求
描述评测硬件和操作系统信息，如下字段必须存在
|  字段                       | 值（举例）       |
|-----------------------------|-----------------|
| accelerator_memory_capacity |                 |
| accelerator_name            |  Qualcomm Hexagon 690 Processor: Hexagon Vector Accelerator (HVX)  |
| accelerators_per_node       |  1              |
| host_memory_capacity        |  6 GB           |
| host_processor_core_count   |  8              |
| host_processor_name         |  Qualcomm Kryo485   |
| host_processors_per_node    |  1              |
| host_storage_capacity       |  32 GB          |
| host_storage_type           |  UFS            |
| number_of_nodes             |  1              |
| operating_system            |  Android 9.0    |
| software_stack              |  PaddleLite 2.8， NDK 17.0 |
| submitter                   |  Baidu          |
| hardware_name               |  某厂商机型      |
| hardware_type               |  mobile         |

### 每个模型code目录中README.md内容要求
- 数据说明
    - 使用的数据集下载方式说明
    - 数据的预处理或后处理的逻辑说明
    - 自测精度、各性能评测及量化校准时分别使用的数据范围及遍历顺序说明。即分别使用了数据集中的哪些数据，以及这些数据使用时按照什么顺序输入

- 模型说明
    - 使用的模型下载方式、模型结构和参数说明，及相比参考模型是否有改变
    - 如果评测要使用量化模型，则需说明量化方法，要求提供量化方法的代码及可执行命令，能生成量化模型用于评测

- 使用的其它优化方法说明，例如：数据重排序等

- 运行代码的步骤，需给出可执行命令，使其能够执行推理，包括：
    - 在某个特定的硬件环境下，安装所需的相关软件步骤
    - 准备数据、模型、模型优化及相关校验等步骤
    - 分别执行精度、各性能评测的推理步骤，产生评测日志

### 每个模型log目录中日志格式要求
执行精度、各性能评测的推理后，产生的评测日志及格式要求如下：

#### 精度评测日志：accuracy_check.log
```
- AI-Rank-log 1558631910.424 load_data, checksum:xxxxxxxxxx
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
- `load_data`：显示所使用数据集的签名
- `test_begin`：测试开始
- `test_finish`：测试结束
- `sampleid`：某个sample的文件名
- `result`：推理结果是否正确，true正确，false错误
- `total_accuracy`：整个推理的准确率

#### 延迟评测日志：latency.log
```
- AI-Rank-log 1558631910.424 load_data, checksum:xxxxxxxxxx
- AI-Rank-log 1558631910.424 test_begin
- AI-Rank-log 1558631910.424 latency_case1_latency:20ms
- AI-Rank-log 1558631910.424 latency_case2_latency:21ms
- AI-Rank-log 1558631910.424 latency_case3_latency:19ms
...
- AI-Rank-log 1558631910.424 latency_case1000_latency:22ms
- AI-Rank-log 1558631910.424 90th_percentile_latency:20ms, min_latency:15ms, max_latency:25ms
- AI-Rank-log 1558631910.424 test_end
```
说明：
- 每行以`AI-Rank-log`开始，后接时间戳
- `load_data`：显示所使用数据集的签名
- `test_begin`：测试开始
- `test_finish`：测试结束
- `latency_caseN_latency`：第N次推理延迟
- `90th_percentile_latency`：取90分位的延迟时间
- `min_latency`：单次推理最大延迟时间
- `max_latency`：单次推理最小延迟时间

#### 离线吞吐评测日志：offline_ips.log
```
- AI-Rank-log 1558631910.424 load_data, checksum:xxxxxxxxxx
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
- `load_data`：显示所使用数据集的签名
- `test_begin`：测试开始
- `test_finish`：测试结束
- `warmup_begin`：预热开始
- `warmup_finish`：预热结束
- `warmup_samples`：预热samples数量
- `total_accuracy`：以完成训练的sample，准确率
- `total_samples_cnt`：已完成的sample数量
- `avg_ips`：`total_samples_cnt` / 所有时间

### 最大并发推理量日志：max_qps_max_memory_use.log
注意：以下只是某一特定并发量的日志格式，提交者提交时，只需提交最大并发量的日志即可。
```
- AI-Rank-log 1558631910.424 load_data, checksum:xxxxxxxxxx
- AI-Rank-log 1558631910.424 test_begin
- AI-Rank-log 1558631910.424 samples_cnt_each_case:100
- AI-Rank-log 1558631912.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:2
- AI-Rank-log 1558631913.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:4
- AI-Rank-log 1558631914.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:6
- AI-Rank-log 1558631915.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:8
- AI-Rank-log 1558631916.424 total_accuracy:0.95943342, max_latency:50ms, total_samples_cnt:10
- AI-Rank-log 1558631910.424 test_end
```
说明：
- 每行以`AI-Rank-log`开始，后接时间戳
- `load_data`：显示所使用数据集的签名
- `test_begin`：测试开始
- `test_finish`：测试结束
- `samples_cnt_each_case`：每次发起预测的samples数量
- `total_accuracy`：已完成测试的所有样本的准确率
- `max_latency`：已完成测试的所有样本的最大延迟
- `total_samples_cnt`：已完成测试的所有样本数量

### summary_metrics.json 结果汇总文件
此文件不在提交范围中。评测日志满足如上要求后，评审专家组会使用工具从日志中生成本次结果汇总文件，放在system目录下，格式为：
|  模型  | 硬件 | 架构(armv7/armv8) | 时延（ms） | 离线吞吐(samples/sec) | 最大并发推理量(samples/sec) |
|--------------|--------------|--------------|--------------|--------------|--------------|
|      -       |      -       |      -       |      -       |      -       |      -       |
