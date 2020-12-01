export CUDA_VISIBLE_DEVICES=6,7,8,9
PORT=29502 ./tools/dist_train.sh configs/fcn/fcn_r50-d8_769x769_40k_torchvision_bdd100k.py 4 --work-dir ./out/fcn_r50-d8_769x769_40k_torchvision/