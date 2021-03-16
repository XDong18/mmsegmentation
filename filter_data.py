import os
import os.path as osp

img_dir = '/shared/xudongliu/bdd100k/100k/train'
label_dir = '/shared/xudongliu/bdd100k/labels/lane/lane_train'
temp_img_dir  = '/shared/xudongliu/bdd100k/100k/train_else/'
label_names = os.listdir(label_dir)

for fn in os.listdir(img_dir):
    name = fn.split('.')[0]+'.png'
    if name not in label_names:
        cmd = 'mv ' + osp.join(img_dir, fn) + ' ' + temp_img_dir
        print('caught ', fn)
        # exit()
    

