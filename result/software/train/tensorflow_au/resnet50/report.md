# NGC TensorFlow ResNet50V1.5 性能测试

此处给出了基于 [NGC TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5) 实现的 ResNet50V1.5 任务的详细复现流程，包括环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [NGC TensorFlow ResNet50V1.5 性能测试](#ngc-tensorflow-resnet50v15-性能测试)
  - [一、环境搭建](#一环境搭建)
  - [二、测试步骤](#二测试步骤)
    - [1.单卡Time2Train及吞吐测试](#1单卡time2train及吞吐测试)
    - [2.单卡准确率测试](#2单卡准确率测试)
  - [三、日志数据](#三日志数据)

## 一、环境搭建

我们遵循了 NGC TensorFlow 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quick-start-guide) 教程成功搭建了测试环境，主要过程如下：

- 下载NGC TensorFlow repo,并进入目录

   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples
   cd DeepLearningExamples/TensorFlow/Classification/ConvNets
   # 本次测试是在如下版本下完成的：
   git checkout 99b1c898cead5603c945721162270c2fe077b4a2
   ```

- 制作Docker镜像

   ```bash
   docker build . -t nvidia_rn50_tf
   ```

- 启动Docker

   ```bash
   # 假设制作好的TF_Record数据放在<path to tfrecords data>目录下
   nvidia-docker run --rm -it -v <path to tfrecords data>:/data/tfrecords --ipc=host nvidia_rn50_tf
   ```

## 二、测试步骤

### 1.单卡Time2Train及吞吐测试

NGC公布的数据为AMP下90个epoch精度可达76.99%，但实际测试90个epoch后精度为75.97%。在120个epoch后精度可以达到AI-Rank要求的76.10%，我们使用如下脚本测试：

```
    python3 main.py --arch=resnet50 \
      --mode=train_and_evaluate --iter_unit=epoch --num_iter=120 \
      --batch_size=256 --warmup_steps=100 --use_cosine --label_smoothing 0.1 \
      --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05 \
      --use_tf_amp --use_static_loss_scaling --loss_scale 128 \
      --data_dir=/data/tfrecords --data_idx_dir=/data/dali_idx \
      --results_dir=/workspace/rn50v15_tf/results --weight_init=fan_in
```

### 2.单卡准确率测试

根据NGC公布的数据，250个epoch基本可以达到最佳训练效果，我们使用如下脚本测试：

```
    python3 main.py --arch=resnet50 \
      --mode=train_and_evaluate --iter_unit=epoch --num_iter=250 \
      --batch_size=256 --warmup_steps=100 --use_cosine --label_smoothing 0.1 \
      --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05 \
      --use_tf_amp --use_static_loss_scaling --loss_scale 128 \
      --data_dir=/data/tfrecords --data_idx_dir=/data/dali_idx \
      --results_dir=/workspace/rn50v15_tf/results --weight_init=fan_in
```

## 三、日志数据
- [单卡Time2Train及吞吐测试日志](./logs/1gpu_time2train_ips.log)
- [单卡准确率测试](./logs/1gpu_accuracy.log)

通过以上日志分析，TensorFlow经过137,335秒的训练完成了90个epoch的训练，训练精度（即`val.top1`)达到76.63 %，训练吞吐（即`train.compute_ips`）达到859.24img/s。
