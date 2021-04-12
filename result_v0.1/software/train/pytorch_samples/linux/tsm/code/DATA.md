# Kinetics-400 数据准备

- [数据集介绍](#数据集介绍)
- [下载video数据](#下载video数据)
- [提取frames数据](#提取frames数据)
- [数据预处理逻辑](#数据预处理逻辑)
- [模型说明](#模型说明)


---


## 数据集介绍

Kinetics-400是视频领域benchmark常用数据集，详细介绍可以参考其官方网站[Kinetics](https://deepmind.com/research/open-source/kinetics)。下载方式可参考官方地址[ActivityNet](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)，使用其提供的下载脚本下载数据集。

## 下载video数据

考虑到K400数据集下载困难的问题，我们将mp4格式的视频数据以zip包的形式上传到了百度云，下载方式如下：

- 下载[train_link.list](https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list)和[val_link.list](https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list)文件链接

编写下载脚本`download.sh`如下:
```bash
file=$1

while read line 
do
  wget "$line"
done <$file
```

下载训练集命令：
```bash
bash download.sh train_link.list
```

下载验证集命令:
```bash
bash download.sh val_link.list
```


|类别 | 数据条数  | list文件 |
| :------: | :----------: | :----: |
|训练集 | 234619  |  [train.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train.list)|
|验证集 | 19761 |  [val.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list)|


- 下载后自行解压，并将数据路径添加到相应的list文件中。

- 由于部分视频原始链接失效，数据有部分缺失，全部文件大概需要135G左右的存储空间，PaddleVideo使用的也是这份数据。

- 训练实用训练集数据，验证和测试均实用验证集数据，训练时数据顺序随机，测试时数据按list顺序。



## 提取frames数据
为了加速网络的训练过程，我们首先对视频文件（K400视频文件为mp4格式）提取帧 (frames)。相对于直接通过视频文件进行网络训练的方式，frames的方式能够极大加快网络训练的速度。

输入如下命令，即可提取K400视频文件的frames

```python
python extract_rawframes.py ./videos/ ./rawframes/ --level 2 --ext mp4
```

视频文件frames提取完成后，会存储在指定的`./rawframes`路径下，大小约为2T左右。

|类别 | 数据条数  | list文件 |
| :------: | :----------: | :----: |
|训练集 | 234619  |  [train_frames.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train_frames.list)|
|验证集 | 19761 |  [val_frames.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val_frames.list)|

## 数据预处理逻辑
### 训练
- GroupMultiScaleCrop: 多尺度裁剪，以[1, .875, .75, .66]比例缩放至不同的尺寸后，从固定的几个位置随机选取一个位置进行裁剪，输出尺寸为224；
- GroupRandomHorizontalFlip: 以0.5的概率随机水平翻转；
- GroupNormalize: 归一化，均值为[0.485, 0.456, 0.406]，方差为[0.229, 0.224, 0.225]。

### 测试
- GroupScale: 缩放至256大小；
- GroupCenterCrop: 中心裁剪至224大小；
- GroupNormalize: 归一化，均值为[0.485, 0.456, 0.406]，方差为[0.229, 0.224, 0.225]。


## 模型说明

### 初始化
使用ImageNet预训练模型进行参数初始化，fc层的weight实用均值为0，方差为0.001的正态分布初始化，bias初始化为0。

### 模型结构
ResNet50，bottleneck第一个conv层之前添加temporal_shift操作

### 优化器
| 优化器 | Momentum |
| :--: | :--: |
|momentum | 0.9 |
|weight_decay | 1e-4 | 
|epoch | 50 |
|总bs | 128 |
|lr | 0.02 |
|lr_decay_ratio | 0.1 |
|lr_decay_step | 20 40 |

### Loss函数
cross entropy损失

### 精度评估方法和频率
top1和top5，每训练完一次进行评估

