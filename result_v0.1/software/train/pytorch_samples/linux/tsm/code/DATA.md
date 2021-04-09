# Kinetics-400 数据准备

- [数据集介绍](#数据集介绍)
- [下载video数据](#下载video数据)
- [提取frames数据](#提取frames数据)

---


## 数据集介绍

Kinetics-400是视频领域benchmark常用数据集，详细介绍可以参考其官方网站[Kinetics](https://deepmind.com/research/open-source/kinetics)。下载方式可参考官方地址[ActivityNet](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)，使用其提供的下载脚本下载数据集。

## 下载video数据

考虑到K400数据集下载困难的问题，我们将mp4格式的视频数据以zip包的形式上传到了百度网盘，用户可根据需要自行下载。

**网盘链接**：https://pan.baidu.com/s/1S_CGBjWOUAuxL_cCX5kMPg
**提取码**：ppvi

|类别 | 数据条数  | list文件 |
| :------: | :----------: | :----: |
|训练集 | 234619  |  [train.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train.list)|
|验证集 | 19761 |  [val.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list)|


- 由于部分视频原始链接失效，数据有部分缺失，全部文件大概需要135G左右的存储空间，PaddleVideo使用的也是这份数据。


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
