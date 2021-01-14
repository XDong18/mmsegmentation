export CUDA_VISIBLE_DEVICES=8,9,2,6
PORT=29502 ./tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_bdd100k.py 4 --work-dir ./out/deeplabv3plus_r50-d8_769x769_80k_bdd100k/