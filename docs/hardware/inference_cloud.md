# AI-Rank 硬件能力评测 - 云端推理方向
## 测试方法
针对云端推理的硬件性能测试，AI-Rank采用固定模型、固定参数、固定测试样本集的方法，在此前提下综合评估被测云端设备的性能、能耗等指标。模型和数据集都采用业界有影响力的，可公开获得的版本，以确保测试的权威性和公平性。

## 模型、参数和数据集
以下3个模型是云端应用场景下，最为频繁使用的推理模型，我们选取以下3个模型，作为验证硬件性能的“测试”程序：
|模型名称 | 应用领域 | 入选理由 |
|----------|---------|---------|
|resnet50 | 图像分类 | - |
|bert | 语义表示 | - |
|transformer | 机器翻译 | - |

为进一步减少软件差异带来的性能影响，最大程度保证公平性，在本任务进行测试时，上述模型的实现需要保证与以下公开模型结构一致，并使用相应的参数集、测试数据集：

|框架 |模型名称 | 模型链接 | 参数集 | 测试数据集 | 
|----------|----------|---------|---------|---------|
| Paddle |resnet50 | - | - |- |
| Paddle |bert | - | - |- |
| Paddle |transformer | - | - |- |
| TensorFlow |resnet50 | - | - |- |
| TensorFlow |bert | - | - |- |
| TensorFlow |transformer | - | - |- |

## 测试环节
测试主要包含两个环节：
- 准确率验证，完成验证数据集所有samples的推理，计算推理准确率。准确率必须不低于如下约束：

|模型名称 | 应用领域 | 准确率约束 | 
|---|---|---|
|resnet50 | 图像分类 | - |
|bert | 语义表示 | - |
|transformer | 机器翻译 | - |

- 指标计算，计算方法下文给出。

## 评价指标

- 离线吞吐：单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。
    - 测试方法：将验证数据集一次性，全部提供给推理程序，推理程序并发推理。计算其整体吞吐速率。

- 在线吞吐：在延迟时间不高于约束值前提下，单位时间内，能够推理的样本数量。单位：samples/sec(样本数/秒)。
    - 测试方法：部署推理服务器，使用测试机模拟client并发请求推理结果，每个请求发送1个sample。逐步增大并发数量，直到响应延迟达到约束延迟时间为止。持续保持该并发量，确保延迟始终不高于约束延迟时间，否则下调并发量。找到一个稳定最大并发值，使得延迟不高于约束延迟时间。在该并发量下，每秒完成推理的样本数平均值，即在线吞吐。
    - 不同模型的约束延迟时间值如下表：

|模型名称 | 应用领域 | 延迟约束 |
|---|---|---|
|resnet50 | 图像分类 | 15 ms |
|bert | 语义表示 | 130 ms |
|transformer | 机器翻译 | - |

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
