# AI-Rank 评测说明 - SSD_MobilenetV3_large(320x320)

## 1. 数据说明
使用 2017 COCO 验证集，具体下载方式说明如下：
* [2017 COCO Validation Images](http://images.cocodataset.org/zips/val2017.zip): 778MB
* [2017 COCO Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip): 241MB，此处评测我们只需要 `instances_val2017.json` 这一个文件
* [2017 COCO label map](https://ai-rank.bj.bcebos.com/coco2017_labelmap.txt?authorization=bce-auth-v1/d0d94402f8e14d64a1695f0bd1e4926a/2021-04-06T08%3A58%3A42Z/-1/host/7ebf837b1999681d4495b391a993d3985447429047a7434cb5bb2564817fccbd)：一个 `.txt` 文件，共 91 行，含有 80 个有效 categories。

```
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip
wget https://ai-rank.bj.bcebos.com/coco2017_labelmap.txt?authorization=bce-auth-v1/d0d94402f8e14d64a1695f0bd1e4926a/2021-04-06T08%3A58%3A42Z/-1/host/7ebf837b1999681d4495b391a993d3985447429047a7434cb5bb2564817fccbd
```

### 处理数据集
2017 COCO 验证集共 4K 张图片，约 1GB 存储空间，因此即使在内存受限的移动设备上评测检测效果，也不需要额外筛选图片，因此使用全集。

首先，执行如下命令：
```
cd /WORK
git clone https://github.com/zhaoyang-star/tensorflow.git
cd tensorflow
bazel run //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:preprocess_coco_minival -- \
  --images_folder=/WORK/val2017 \
  --instances_file=/WORK/annotations/instances_val2017.json \
  --output_folder=/WORK/2017_COCO_Minival
```

经过如上调用预处理工具 `preprocess_coco_minival` Python 脚本，会在指定的输出路径下生成:
* `images`: 2017 COCO Validation images 中的全集，`images/*.jpg` 文件
* `ground_truth.pb`: 子集对应的 ground truth 数据
```
demo@tflite:/WORK$ ls 2017_COCO_Minival/
ground_truth.pb  images
```

### 数据使用
在移动端设备上执行推理时，图片按照其字母 `a-z` 排序和数字 `0-9` 排序（底层实现为：先将所有图片名字转为 `std::string` 类型，然后调用 `std::sort` 排序），依次输入到模型中。


## 2. 模型说明

模型 | MD5 | 备注
---|---|---
[tflite 格式的 ssd_mobilenet_v3_large](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz) | 7630ae1167de533b2dfcdd1a7f21eb64 |

模型只有一个输入 Tensor，输入维度 [1,320,320,3]，数据类型 `uint8`，数据排布 `NHWC`.
模型有 4 个输出 Tensor，具体信息如下表：

Outputs 索引 | 名称 | 描述
---|---|---|
0 | 坐标 | [10][4] 多维数组，每一个元素由 0 到1 之间的浮点数，内部数组表示了矩形边框的 [top, left, bottom, right]
1 | 类型 | 10个整型元素组成的数组（输出为浮点型值），每一个元素代表标签文件中的索引
2 | 分数 | 10个整型元素组成的数组，元素值为 0 至 1 之间的浮点数，代表检测到的类型
3 | 检测到的物体和数量	 | 长度为1的数组，元素为检测到的总数


## 3. 使用的其它优化方法说明
无。


## 4. 编译代码
在某个特定的硬件环境下，安装所需的相关软件步骤
准备数据、模型、模型优化及相关校验等步骤
分别执行精度、各性能评测的推理步骤，产生评测日志

编译目标检测评测工具：
```
cd /WORK
git clone https://github.com/zhaoyang-star/tensorflow.git
cd tensorflow

# 安装 bazel
cd "/usr/local/lib/bazel/bin" && curl -LO https://releases.bazel.build/3.7.2/release/bazel-3.7.2-linux-x86_64 && chmod +x bazel-3.7.2-linux-x86_64 && cd -

# [option] 编译 armv7 版本的评测工具
bazel build -c opt --config=android_arm  //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval

# [option] 编译 armv8 版本的评测工具
bazel build -c opt --config=android_arm64  //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval
```
经执行如上命令后，会生成二进制可执行文件：`bazel-bin/tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval`。为了避免访问受限，可以拷贝可执行文件到另一个目录下：
```
cp bazel-bin/tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval /WORK
```
备注：armv7/v8 产物目录名称均为 `bazel-bin`，为了区分二者建议：在编译完 armv8 版本的评测工具后，将 `bazel-bin` 重命名为 `bazel-bin-v8`；在编译完 armv7 版本的评测工具后，将 `bazel-bin` 重命名为 `bazel-bin-v7`。这样，两者就不会重复命名。

## 5. 上传数据到移动端设备
将评测二进制文件、模型文件、数据集一并上传到移动设备：
```
cd /WORK
# 创建测试目录
adb shell mkdir /data/local/tmp/AI-RANK/tf_object_detction
# 上传可执行文件
adb push run_eval /data/local/tmp/AI-RANK/tf_object_detction
adb shell chmod +x /data/local/tmp/AI-RANK/tf_object_detction/run_eval
# 上传模型文件
adb push ssd_mobilenet_v3_large_coco_2020_01_14/model.tflite /data/local/tmp/AI-RANK/tf_object_detction
# 上传验证集
tar xf val2017.tar val2017
adb push val2017.tar /data/local/tmp/AI-RANK/tf_object_detction
# 上传真值
adb push 2017_COCO_Minival/ground_truth.pb /data/local/tmp/AI-RANK/tf_object_detction
# 上传 label 文件
adb push labelmap_2017.txt /data/local/tmp/AI-RANK/tf_object_detction
```


## 6. 运行
```
adb shell /data/local/tmp/AI-RANK/tf_object_detction/run_eval \
  --model_file=/data/local/tmp/AI-RANK/tf_object_detction/model.tflite \
  --ground_truth_images=/data/local/tmp/AI-RANK/tf_object_detction/val2017.tar \
  --ground_truth_proto=/data/local/tmp/AI-RANK/tf_object_detction/ground_truth.pb \
  --model_output_labels=/data/local/tmp/AI-RANK/tf_object_detction/labelmap_2017.txt \
  --delegate=xnnpack \
  2>&1 | tee log
```

`run_eval` 可执行文件参数介绍。
- 必须提供参数如下：
    *  `model_file` : `string` \
    TFlite 模型文件路径
    *  `ground_truth_images`: `string` \
    测试图片集路径

    *   `ground_truth_proto`: `string` \
    测试图片集的真值路径

    *   `model_output_labels`: `string` \
    coco2017验证集的label文本文件路径

- 可选参数如下:

    *   `debug_mode`: `int`  (default=0) \
    打开debug_mode，可以获得图片预处理耗时和每张图片具体的预测结果

    *   `num_interpreter_threads`: `int` (default=1) \
    预测的线程数

    *   `delegate`: `string` (default='cpu')\
    推理预测的硬件设备选择，默认是`CPU`，可选值有：`GPU`、`xnnpack`、`hexagon`、`nnapi`等
    更多可用的参数介绍，请见code目录下的[README](https://github.com/zhaoyang-star/tensorflow/blob/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/README.md)

## 7. 性能及精度评测结果
| 模型  | 硬件 |架构(armv7/armv8) | 时延（ms） |  mAP    |  
|------------------|--------------|--------------|--------------|------------|
| SSD-MobileNetV3  |     cpu      |     armv7    |    60.554    |    0.254   |
| SSD-MobileNetV3  |     gpu      |     armv7    |    38.108    |    0.254   |
| SSD-MobileNetV3  |     cpu      |     armv8    |    44.469    |    0.254   |
| SSD-MobileNetV3  |     gpu      |     armv8    |    30.082    |    0.254   |
