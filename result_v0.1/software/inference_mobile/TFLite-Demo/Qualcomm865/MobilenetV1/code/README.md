# AI-Rank 评测说明 - MobilenetV1(224x224)

## 1. 数据说明
2012 ILSVRC 验证集共 5K 张图片，约 6GB 存储空间，占用空间比较大，故从中筛选1K张图片进行验证。
筛选方法：从1000类分类数据集中，各挑选一张图片作为验证图片进行测试。
筛选方法：[请见脚本](./choose_1k.py)

[ILSVRC 2012 image classification task](http://www.image-net.org/challenges/LSVRC/2012/)包含5K 张图片，可在此链接下载。

ILSVRC 2012 1k 验证集图片下载方式：
```
wget https://ai-rank.bj.bcebos.com/ILSVRC2012_1000_cls.zip?authorization=bce-auth-v1/d0d94402f8e14d64a1695f0bd1e4926a/2021-04-06T08%3A58%3A15Z/-1/host/73fb63b35c284154f6a7e456d8550210c917b3efd69767b802a15511bfe0acae
```
ILSVRC 2012 1000类label 文档下载方式：
```
wget https://ai-rank.bj.bcebos.com/imagenet1k_label_list.txt?authorization=bce-auth-v1/d0d94402f8e14d64a1695f0bd1e4926a/2021-04-06T08%3A56%3A50Z/-1/host/56ba5989ffada9317e83395ff4d3a55c2b24a62b6c70c2e074aeea236edd1a0a
```

### 数据使用
在移动端设备上执行推理时，图片按照其字母 `a-z` 排序和数字 `0-9` 排序（底层实现为：先将所有图片名字转为 `std::string` 类型，然后调用 `std::sort` 排序），依次输入到模型中。


## 2. 模型说明

模型 | MD5 | 备注
---|---|---
[tflite 格式的 mobilenet_v1](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | bf077f18f3b282945f03f5109bd0eb90 |

- 模型只有一个输入 Tensor，输入维度 [1,240,240,3]，数据类型 `uint8`，数据排布 `NHWC`.
模型有 1 个输出 Tensor, 输出维度[1,1001]，数据类型`float`

- 模型下载后，解压模型文件，可获取已转换好的TFLite 模型文件

## 3. 使用的其它优化方法说明
无。


## 4. 编译代码
在某个特定的硬件环境下，安装所需的相关软件步骤
准备数据、模型、模型优化及相关校验等步骤
分别执行精度、各性能评测的推理步骤，产生评测日志

编译分类评测工具：
```
cd /WORK
git clone https://github.com/zhaoyang-star/tensorflow.git
cd tensorflow
# 安装 bazel
cd "/usr/local/lib/bazel/bin" && curl -LO https://releases.bazel.build/3.7.2/release/bazel-3.7.2-linux-x86_64 && chmod +x bazel-3.7.2-linux-x86_64 && cd -
# [option] 编译 armv7 版本的评测工具
bazel build -c opt --config=android_arm  //tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval
# [option] 编译 armv8 版本的评测工具
bazel build -c opt --config=android_arm64  //tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval
```
经执行如上命令后，会生成二进制可执行文件：`bazel-bin/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/run_eval`。


备注：armv7/v8产物目录名称均为bazel-bin，为了区分二者建议：在编译完armv8 版本的评测工具后，将bazel-bin重命名为bazel-bin-v8；在编译完armv7 版本的评测工具后，将bazel-bin重命名为bazel-bin-v7。这样，两者就不会重复命名


## 5. 上传数据到移动端设备
将评测二进制文件、模型文件、数据集一并上传到移动设备：

```
cd /WORK
# 在手机上新建目录
adb shell mkdir /data/local/tmp/AI-RANK/tf_lite_image_classify
# push 可执行文件
adb push bazel-bin-v8/tensorflow/lite/tools/evaluation/tasksrun_eval /data/local/tmp/AI-RANK/tf_lite_image_classify
# 赋值可执行文件权限
adb shell chmod +x /data/local/tmp/AI-RANK/tf_lite_image_classify/run_eval
# push 模型至手机
adb push mobilenet_v1_1.0_224.tflite /data/local/tmp/AI-RANK/tf_lite_image_classify
# puss 下载完成的1K验证集至手机
adb push ILSVRC2012_1000_cls.zip /data/local/tmp/AI-RANK/tf_lite_image_classify
adb push ILSVRC2012_1000_cls /data/local/tmp/AI-RANK/tf_lite_image_classify
# puss 下载完成的imagenet1k_label_list.txt至手机
adb push imagenet1k_label_list.txt /data/local/tmp/AI-RANK/tf_lite_image_classify
```

## 6. 运行
```
adb shell /data/local/tmp/AI-RANK/tf_lite_image_classify/run_eval \
  --model_file=/data/local/tmp/AI-RANK/tf_lite_image_classify/mobilenet_v1_1.0_224.tflite
  --ground_truth_images_path=/data/local/tmp/AI-RANK/tf_lite_image_classify/ILSVRC2012_1000_cls.zip
  --ground_truth_labels=/data/local/tmp/AI-RANK/tf_lite_image_classify/ILSVRC2012_1000_cls/val_list_1k.txt
  --model_output_labels=/data/local/tmp/AI-RANK/tf_lite_image_classify/imagenet1k_label_list.txt
  --delegate=xnnpack \
  --num_threads=1 \
  --topk=5>accuracy_check_latency.log
```
run_eval 可执行文件参数介绍。
- 必须提供参数如下：
    *  `model_file` : `string` \
    TFlite 模型文件路径
    *  `ground_truth_images_path`: `string` \
    测试图片集路径

    *   `ground_truth_labels`: `string` \
    测试图片集的真值结果（label）文本路径

    *   `model_output_labels`: `string` \
    1000类分类label文本路径

- 可选参数如下:

    *   `debug_mode`: `int`  (default=0) \
    打开debug_mode，可以获得图片预处理耗时和每张图片具体的预测结果

    *   `num_threads`: `int` (default=4) \
    预测的线程数

    *   `delegate`: `string` (default='cpu')\
    推理预测的硬件设备选择，默认是`CPU`，可选值有：`GPU`、`xnnpack`、`hexagon`、`nnapi`等
    更多可用的参数介绍，请见code目录下的[README](https://github.com/zhaoyang-star/tensorflow/blob/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/README.md)

## 7. 性能及精度评测结果
| 模型              |    硬件      | 架构(armv7/armv8) | 时延（ms） |  Top-1（%）|
|------------------|--------------|--------------|--------------|------------|
| MobilenetV1      |     cpu      |     armv7    |    31.278    |    72.8%   |
| MobilenetV1      |     gpu      |     armv7    |    8.003     |    72.8%   |
| MobilenetV1      |     cpu      |     armv8    |    29.478    |    72.8%   |
| MobilenetV1      |     gpu      |     armv8    |    7.494     |    72.8%   |

