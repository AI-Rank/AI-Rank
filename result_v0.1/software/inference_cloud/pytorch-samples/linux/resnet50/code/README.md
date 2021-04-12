# AI-Rank 评测说明

## 使用说明

该代码修改自[pytorch example(commit 49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de)](https://github.com/pytorch/examples/tree/master/imagenet)。测试时请使用本目录下imagenet文件夹内的代码。

## 安装

- 安装 PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r imagenet/requirements.txt`
- 从 http://www.image-net.org/ 下载ImageNet数据集
    - 把脚本放到验证集图像同一目录内然后运行[脚本](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## 运行命令

cd imagenet  
python main.py -a resnet50 -j 8 -b 64 -e --pretrained --gpu 0 [imagenet-folder with train and val folder]

## 参数说明

```
  --arch ARCH, -a ARCH  模型名: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (默认: resnet18)
  -j N, --workers N     数据加载并发数 (默认: 4)
  -b N, --batch-size N  批尺寸 (默认: 256)
  -e, --evaluate        使用验证集数据
  --pretrained          使用预训练模型
  --gpu GPU             使用的GPU ID.
  --warmup              预热次数(默认：16).
```

## 评测结果

|  模型  | 离线吞吐(samples/sec)  | 在线吞吐(samples/sec) |
|--------------|--------------|--------------|
|   Resnet50   |    364       |              |