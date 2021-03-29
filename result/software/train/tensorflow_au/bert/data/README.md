<!-- omit in toc -->
# NGC TensorFlow Bert 数据预处理

此处给出了基于 [NGC TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) 实现的 Bert Base Pre-Training 任务的详细数据下载、预处理的流程。

<!-- omit in toc -->
## 目录
- [一、环境搭建](#一环境搭建)
- [二、数据处理](#二数据处理)

## 一、环境搭建

NGC TensorFlow 的代码仓库提供了自动构建 Docker 镜像的的 [shell 脚本](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/docker/build.sh)，可参考[此处](../script/README.md)创建并启动容器：

- **镜像版本**: `nvcr.io/nvidia/tensorflow:20.06-tf1-py3`
- **TensorFlow 版本**: `1.15.2+nv`
- **CUDA 版本**: `11.0`
- **cuDnn 版本**: `8.0.1`

## 二、数据处理

我们遵循了 NGC TensorFlow 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT#quick-start-guide) 教程成功搭建了测试环境。

- **数据下载**

  NGC TensorFlow 提供单独的数据下载和预处理脚本 [data/create_datasets_from_start.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/data/create_datasets_from_start.sh)。在容器中执行如下命令，可以下载和制作 `wikicorpus_en` 的 tfrecord 数据集。

  ```bash
  bash data/create_datasets_from_start.sh wiki_only
  ```

  由于数据集比较大，且容易受网速的影响，上述命令执行时间较长。因此，为了更方便复现竞品的性能数据，我们提供了已经处理好的 tfrecord 格式[样本数据集](https://bert-data.bj.bcebos.com/benchmark_sample%2Ftfrecord.tar.gz)。

  下载后的数据集需要放到容器中`/workspace/bert/data/`目录下，并修改[scripts/run_pretraining_lamb_phase1.sh](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/scripts/run_pretraining_lamb_phase1.sh#L81)的第81行的数据集路径,如：

  ```bash
  # 解压数据集
  tar -xzvf benchmark_sample_tfrecord.tar.gz
  # 放到 data/目录下
  mv benchmark_sample_tfrecord bert/data/tfrecord
  # 修改 run_pretraining_lamb_phase1 L81 行数据集路径
  INPUT_FILES="$DATA_DIR/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/training"
  EVAL_FILES="$DATA_DIR/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/wikicorpus_en/test"
  ```