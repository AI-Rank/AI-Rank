# open-mmlab/mmsegmentation HRNet_W18 性能测试

测试基于[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)公开的文档及代码。

## 目录
- [一、环境安装](#一、环境安装)
- [二、数据准备](#二、数据准备)
- [三、训练](#三、训练)
- [四、日志数据](#四、日志数据)
- [五、性能指标](#五、性能指标)


## 一、环境安装
参考官方文档[get_start](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md)进行环境的安装。其基于anaconda进行安装，anaconda的安装及使用请参考其[官方网站](https://www.anaconda.com/)

如下是安装的简要流程

1. 安装torch和torchvision
```
pip install torch==1.7.1 torchvision==0.8.2
```

2. mmcv的安装
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.1/index.html
```
**note:** 根据实际情况选择相应的[版本](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation)

3. mmsegmentation安装
```
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # or "python setup.py develop"
```

## 二、数据准备
测试数据集基于CityScapes，请参考官方说明进行相应的[数据集准备](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md)。

## 三、训练

修改评估间隔为1000，将`configs/_base_/schedules/schedule_80k.py `第8、9行修改如下：
```
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')
```

- 单卡训练

单GPU训练需使用BN, 修改`configs/_base_/models/fcn_hr18.py`第二行如下： 
```
norm_cfg = dict(type='BN', requires_grad=True)
```

训练命令
```
export CUDA_VISIBLE_DEVICES=0
python tools/train.py configs/hrnet/fcn_hr18_512x1024_80k_cityscapes.py --gpus 1 --work-dir ./results
```

- 单机多卡

修改`configs/_base_/models/fcn_hr18.py`, 多卡训练需使用SyncBN：
```
norm_cfg = dict(type='SyncBN', requires_grad=True)
```

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./tools/dist_train.sh configs/hrnet/fcn_hr18_512x1024_80k_cityscapes.py 8

```

- fp16训练

在目录`configs/fp16`下新建配置文件`fcn_hr18_512x1024_80k_fp16_cityscapes.py`
并输入如下内容：
```python
_base_ = '../hrnet/fcn_hr18_512x1024_80k_cityscapes.py'
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
```
训练流程与单卡和单机多卡一样，只需对配置文件做出相应的更改。

## 四、日志数据
- [单卡吞吐测试日志](../log/GPUx1_time2train_ips.log)
- [多卡准确率测试日志](../log/GPUx8_time2train_ips.log)
- [单卡吞吐测试日志-fp16](../log/fp16_GPUx1_time2train_ips.log)
- [多卡准确率测试日志-fp16](../log/fp16_GPUx8_time2train_ips.log)

## 五、性能指标

### fp32测试结果

| GPU卡数       | Time2train(sec)  | 吞吐(images/sec) | 准确率(%) | 加速比 |
|------------- |------------------|------------------|----------|-------|
| 1卡          |       -          |      7.00        |     -      |   -  |
| 8卡          |      57574.09    |      20.88       |     78.11  |  2.98 |

**note:** 

mmsegmentation 版本：

tag: v0.11.0

commit id: 0448dec5e4690ced25f8ee63e1a6b0f83f84cac8

### fp16测试结果
| GPU卡数       | Time2train(sec)  | 吞吐(images/sec) | 准确率(%) | 加速比 |
|------------- |------------------|------------------|----------|-------|
| 1卡          |       -          |      5.20        |     -      |   -  |
| 8卡          |      70489.24    |      17.82       |     78.46  |  3.43 |

**note:** 

mmsegmentation 版本：

commit id: 4b9c9a3ad3ee811b763db8672c0c18fd97b2513a
