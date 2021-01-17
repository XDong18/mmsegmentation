export CUDA_VISIBLE_DEVICES=2,3,4,5
PORT=29502 ./tools/dist_train.sh configs/dla/dla34up_80k_new_sbn.py 4 --work-dir ./out/dla34up_80k_new_sbn/