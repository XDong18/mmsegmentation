export CUDA_VISIBLE_DEVICES=1,2
PORT=29502 ./tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_bdd100k.py 2 --work-dir ./out/deeplabv3plus_r50-d8_769x769_80k_bdd100k/