export CUDA_VISIBLE_DEVICES=6,7,8,9
python tools/test.py configs/dla/dla34up_80k_new_sbn.py ./out/dla_model/dla34_bs16_500e.pth --eval mIoU