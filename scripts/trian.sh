export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29501 ./tools/dist_train.sh configs/fcn/fcn_r50-d8_512x512_20k_bdd100k.py 4 --work-dir out/fcn_r50-d8_512x512_20k