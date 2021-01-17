_base_ = [
    '../_base_/models/dla102up.py', '../_base_/datasets/bdd100k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (768, 768)
test_cfg = dict(mode='slide', crop_size=(769, 769), stride=(513, 513))