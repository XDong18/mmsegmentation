export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29502 ./tools/dist_train.sh configs/danet/danet_r50-d8_769x769_40k_bdd100k.py 4 --work-dir out/danet_r50-d8_769x769_40k