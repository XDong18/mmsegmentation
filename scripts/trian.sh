export CUDA_VISIBLE_DEVICES=6,7,8,9
PORT=29507 ./tools/dist_train.sh configs/dla/dla34up_80k_new_sbn.py 4 --work-dir ./out/dla34up_80k_new_sbn/ \
--resume ./out/dla34up_80k_new_sbn/iter_80000.pth