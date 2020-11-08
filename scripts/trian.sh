export CUDA_VISIBLE_DEVICES=5,6,7,8
PORT=29502 ./tools/dist_train.sh configs/emanet/emanet_r50-d8_769x769_80k_bdd100k.py 4 --work-dir out/emanet_r50-d8_769x769_80k