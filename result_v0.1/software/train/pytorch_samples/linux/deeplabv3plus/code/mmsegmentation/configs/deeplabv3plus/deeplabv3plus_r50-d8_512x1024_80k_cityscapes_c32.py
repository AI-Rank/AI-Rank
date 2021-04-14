_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime_c32.py',
    '../_base_/schedules/schedule_80k_c32.py'
]
