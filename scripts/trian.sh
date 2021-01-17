export CUDA_VISIBLE_DEVICES=6,7,8,9
PORT=29507 ./tools/dist_train.sh configs/emanet/emanet_r101-d8_769x769_80k_bdd100k.py 4 --work-dir ./out/emanet_r101-d8_769x769_80k_bdd100k/