export CUDA_VISIBLE_DEVICES=6,7,8,9
PORT=29503 ./tools/dist_test.sh configs/emanet/emanet_r50-d8_769x769_80k_bdd100k.py \
    c_out/emanet_r50-d8_769x769_80k/latest.pth \
    4 --eval mAP