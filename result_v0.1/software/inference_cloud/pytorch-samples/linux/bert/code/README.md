<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Source

https://github.com/huggingface/transformers has been adapted for AI-RANK.

## Installation

pip install -e .  
pip install -r examples/text-classification/requirements.txt

## Benchmark
python run_glue.py --model_name_or_path [path to bert-base-uncased] --task_name sst2 --max_seq_length 128 --output_dir [output path] --do_eval --per_device_eval_batch_size 4 --warmup_times 8 --ai_rank_logging True
