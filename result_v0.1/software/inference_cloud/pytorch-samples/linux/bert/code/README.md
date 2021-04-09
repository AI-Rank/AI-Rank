## Source

https://github.com/huggingface/transformers has been adapted for AI-RANK(we add some log related codes and remove some useless files).

## Installation

cd transfomers  
pip install -e .  
cd question-answering  
pip install -r requirements.txt

## Benchmark
python run_qa.py --model_name_or_path [path to model of bert-large-uncased]  --dataset_name squad --do_eval --per_device_eval_batch_size 32  --max_seq_length 384 --output_dir ./debug_squad/ --ai_rank_logging True --warmup_times 8

## checksum
we calculate md5 on each sample: 'answers' + 'context' + 'id' + 'question + 'title' 

## summary_metrics
|  模型  | 离线吞吐(samples/sec)  | 在线吞吐(samples/sec) |
|--------------|--------------|--------------|
|   Resnet50   |    38       |              |