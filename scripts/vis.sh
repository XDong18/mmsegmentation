export CUDA_VISIBLE_DEVICES=6,7,8,9
python tools/test.py configs/dla/dla34up_80k_new_sbn.py /shared/xudongliu/code/f_server/mmsegmentation/out/dla34up_80k_new_sbn/latest.pth --eval mIoU --show-dir show/dla34up_80k_new_sbn_old_config