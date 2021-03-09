export CUDA_VISIBLE_DEVICES=4,5
python tools/test.py configs/dla/dla102up_80k_new.py ./out/fcn_r101-d8_769x769_40k_bs16_bdd100k/latest.pth \
--eval mIoU --show-dir ./show/fcn_r101-d8_769x769_40k_bs16_bdd100k