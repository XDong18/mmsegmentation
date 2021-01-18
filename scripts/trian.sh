export CUDA_VISIBLE_DEVICES=6,7,8,9
PORT=29507 ./tools/dist_train.sh configs/dla/dla34up_220k_new_sbn.py 4 --work-dir ./out/dla34up_220k_new_sbn/