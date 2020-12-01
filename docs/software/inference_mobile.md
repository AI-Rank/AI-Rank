# AI-Rank 软件能力评测 - 终端推理方向

## 测试方法
AI-Rank针对终端推理框架设置了几个维度的不同指标来评价其综合能力，除常规的性能测试指标外，还考察推理框架对市面上各种常见终端硬件的适配能力，因此本任务包括以下几个维度：
- 框架性能：在固定模型、固定参数及固定测试样本集的条件下，测试不同框架的各项性能指标。模型和数据集都采用业界有影响力的，可公开获得的版本，以确保测试的权威性和公平性。
- 硬件适配能力：选取当前市面上常见的终端硬件，分为必选硬件和可选硬件两类，被测框架需至少完成所有必选硬件的测试任务，除必选硬件外，框架能够在可选硬件上完成的任务数量越多，说明其适配能力越好。
- 能耗：对于终端硬件，能耗是一项重要指标，但目前尚未对该指标设置具体评测方案，会在后续版本中进行设置。

## 硬件环境
基于目前对产业界，科研领域等的调研，AI-Rank指定了如下终端硬件，并不限定操作系统、驱动版本，后续也会随着业界发展动态更新或添加新的环境：
- 必选硬件：
    - 高通855小米9
    - 高通6660 OPPO R11S
    - 高通625 红米6pro
    - iPhone Xr
- 可选硬件：
    - iPhone6s
    - iPhone5s
    - Jetson AGX Xavier
    - 树莓派4B
    - RK3399

## 模型任务及参数和数据集
AI-Rank从深度学习应用最广泛的几大领域中分别选取了若干有影响力的模型，参与方可以根据自己使用的框架按照标准模型结构进行实现：

|模型名称 | 应用领域 | 入选理由|
|---|---|---|
|mobilenetv3 | 图像分类 | - |
|shufflenet_v2 | 图像分类 | - |
|squeezenet_v1.1 | 图像分类 | - |

为进一步减少模型差异带来的性能影响，最大程度保证公平性，在本任务进行测试时，AI-Rank要求固定模型网络结构、参数集、测试数据集。因此AI-Rank提供了基于TensorFlow的模型、完成训练的参数集和测试数据集。
|模型名称 | 模型链接 | 参数集 | 测试数据集 | 
|----------|----------|---------|---------|
|mobilenetv3 | - | - | - |
|shufflenet_v2 | - | - | - |
|squeezenet_v1.1 | - | - | - |

## 测试环节

测试主要包含两个环节：
- 准确率验证，完成验证数据集所有samples的推理，计算推理准确率。准确率必须不低于如下约束：

|模型名称 | 应用领域 | 准确率约束 | 
|---|---|---|
|mobilenetv3 | 图像分类 | - |
|shufflenet_v2 | 图像分类 | - |
|squeezenet_v1.1 | 图像分类 | - |

- 指标计算，计算方法下文给出。

## 评价指标
- 硬件适配能力：该框架能够在给定硬件上（包括可选和必选）完成的任务数占总任务数的比例。本指标主要体现框架对业界常用硬件的适配全面性。指标根据参与者提交的任务个数确定。
- 性能指标：
    - 时延：N次推理，每次1个sample，计算每次推理的用时。时延取N次用时的90分位值。单位：ms（毫秒）
        - 测试方法：取1000个样本，每次推理一个样本，计算该样本推理延迟时间。1000个延迟时间升序排序，取90分位值。
    - 最大并发推理量：在延迟时间不高于约束值前提下，最大支持的一个批次的BatchSize值。
        - 测试方法：部署推理程序到终端，编写测试程序，加载验证数据集，每次加载N个samples。N的数量逐步增大，直到响应延迟达到约束延迟时间为止。持续保持N的值，确保延迟始终不高于约束延迟时间，否则下调N值。找到一个稳定N值，使得延迟不高于约束延迟时间。N值及即最大并发推理量。
        - 不同模型的约束延迟时间值如下表：

|模型名称 | 应用领域 | 延迟约束 | 
|---|---|---|
|mobilenetV3 | 图像分类 | 50 ms |
|shufflenet_v2 | 图像分类 | - |
|squeezenet_v1.1 | 图像分类 | - |

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
|  模型  | 硬件 | 时延（ms） | 最大并发推理量(samples/sec) | 内存占用(MB) |
|--------------|--------------|--------------|--------------|--------------|
|      -       |      -       |      -       |      -       |      -       |

## 目录结构

提交目录结构示例如下：

- system name
    - system information
    - model1-iPhone6s
        - code
        - data
        - log
            - accuracy_check.log
            - latency.log
            - max_qps_max_memory_use.log
        - report
    - model2-RK3399
        - code
        - data
        - log
            - accuracy_check.log
            - latency.log
            - max_qps_max_memory_use.log
        - report
    - summary metrics
    - submitter information
