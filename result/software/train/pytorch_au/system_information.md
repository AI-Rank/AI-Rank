<!-- omit in toc -->

本次所提交的NGC PyTorch数据，主要参考[NGC官网](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch)公开的文档及代码。

- 测试所使用的环境为：
  - 系统：CentOS Linux release 7.5.1804
  - GPU：Tesla V100-SXM2-16GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 38
  - Driver Version: 450.80.02
  - 内存：432 GB

- 镜像及版本信息如下：
  - Docker: nvcr.io/nvidia/pytorch:20.07-py3
  - PyTorch：1.6.0a0+9907a3e
  - CUDA：11
  - cuDNN：8.0.1
  - [ResNet50模型代码](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)
  - [Bert模型代码](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)