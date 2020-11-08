export CUDA_VISIBLE_DEVICES=4,5,6,7
PORT=29503 ./tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_40k_bdd100k.py 4 --work-dir out/deeplabv3plus_r50-d8_769x769_40k