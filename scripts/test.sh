export CUDA_VISIBLE_DEVICES=6,7,8,9
PORT=29503 ./tools/dist_test.sh configs/pspnet/pspnet_r50-d8_769x769_40k_bdd100k.py \
    f_out/pspnet_r50-d8_769x769_40k/latest.pth \
    4 --eval mAP