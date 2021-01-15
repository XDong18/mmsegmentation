export CUDA_VISIBLE_DEVICES=6,7,8,9
PORT=29502 ./tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_bdd100k.py 4 --work-dir ./out/deeplabv3plus_r50-d8_769x769_80k_bdd100k/ \
--resume-from ./out/deeplabv3plus_r50-d8_769x769_80k_bdd100k/latest.pth