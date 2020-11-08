export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29503 ./tools/dist_train.sh configs/fcn/fcn_r50-d8_769x769_40k_bdd100k.py 4 --work-dir out/fcn_r50-d8_769x769_40k