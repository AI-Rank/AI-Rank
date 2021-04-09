## 安装

cd transfomers  
pip install -e .  
cd question-answering  
pip install -r requirements.txt

## 运行
python run_qa.py --model_name_or_path [path to model of bert-large-uncased]  --dataset_name squad --do_eval --per_device_eval_batch_size 32  --max_seq_length 384 --output_dir ./debug_squad/ --ai_rank_logging True --warmup_times 8

## 校验和计算
在每一条输入数据上计算md5值: 'answers' + 'context' + 'id' + 'question + 'title' 

## 评测结果
|  模型  | 离线吞吐(samples/sec)  | 在线吞吐(samples/sec) |
|--------------|--------------|--------------|
|   Resnet50   |    38       |              |