<!-- omit in toc -->
# NGC PyTorch Transformer 性能复现


此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) 实现的 Transformer 的详细复现流程，包括执行环境、PyTorch版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->
## 目录
- [一、环境搭建](#一环境搭建)
- [二、数据准备](#二数据准备)
- [三、测试步骤](#三测试步骤)
  - [1. 单机吞吐测试](#1-单机吞吐测试)
- [四、日志数据](#四日志数据)
- [五、性能数据](#五性能数据)



## 一、环境搭建

我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer#quick-start-guide) 教程搭建了测试环境，主要过程如下：

- **拉取代码**

    ```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples.git 
    cd DeepLearningExamples/PyTorch/Translation/Transformer
    ```

- **构建镜像**

    构建镜像之前，为了使用 apex，我们修改了 [Dockerfile](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/Dockerfile#L19) 打开 apex 相关注释。

    ```bash
    RUN git clone https://github.com/NVIDIA/apex \
     && cd apex \
     && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```

    执行镜像构建以及启动 docker。

    ```bash
    docker build . -t your.repository:transformer
    nvidia-docker run -it --rm --ipc=host your.repository:transformer bash
    ```

    若已经有了处理好的数据，可以使用：

    ```bash
    nvidia-docker run -it --rm --ipc=host -v <path to your preprocessed data>:/data/wmt14_en_de_joined_dict your.repository:transformer bash
    ```

    若数据已经下载好了但是没有处理，可以使用：

    ```bash
    nvidia-docker run -it --rm --ipc=host -v <path to your unprocessed data>:/workspace/translation/examples/translation/orig your.repository:transformer bash
    ```

## 二、数据准备

NGC PyTorch 提供单独的数据下载和预处理脚本，详细的数据处理流程按照[官方文档](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer#quick-start-guide)，可以通过执行以下脚本，完成包括 WMT 数据集的下载以及数据处理步骤：

```bash
scripts/run_preprocessing.sh
```

## 三、测试步骤

为了更准确的测试 NGC PyTorch 的性能数据，我们严格按照官方提供的模型代码配置、启动脚本，进行了的性能测试。

### 1.单机吞吐测试

根据NGC公布的测试方式，我们可以使用一下命令执行训练：

执行单卡训练直接使用 `python` 执行 `train.py` 脚本即可。执行多卡训练需要使用 `python -m torch.distributed.launch --nproc_per_node $N`，其中 `$N` 是使用的 GPU 卡的数目。

```bash
python -m torch.distributed.launch --nproc_per_node 8 /workspace/translation/train.py /data/wmt14_en_de_joined_dict \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens 5120 \
  --seed 1 \
  --fuse-layer-norm \
  --amp \
  --amp-level O2 \
  --save-dir /workspace/checkpoints \
  --distributed-init-method env:// 
```

为了复现 [Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187) 论文中的 BLEU 的结果，NGC 使用混合精度，并设定 `batch_size=5120` 以及 ` lr=6e-4` 在 `DGX-1V (8x V100s 16G)` 上进行训练。如果使用不同的启动配置，有以下两点需要注意：

* 若使用 FP32 进行训练，因 16G 显存限制，需要将 batch_size 设为 2560，并且设置 `--update-freq 2` 以使用梯度累加。
* 若在较少的 GPU 上进行训练，需要将 `--update-freq` 按照对应的比例增大。

举个例子，如果使用 FP32 在 4 卡上进行训练，需要设置 `--update-freq=4`。

## 四、日志数据

- [单机八卡吞吐日志](../logs/transformer_big_gpu8_fp32.log)

通过以上日志分析，PyTorch 在 Transformer 任务上的单机吞吐达到了 **54752.958** `tokens/sec` 。

## 五、性能数据

|               | Time2train(sec) | 吞吐(tokens/sec) | BLEU | 加速比 |
|---------------|-----------------|-----------------|------|-------|
| 1卡           |        -        |     8392.091    |   -  |   -   |
| 8卡           |    77678.071    |    54752.958    |   -  |   -   |
