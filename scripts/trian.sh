export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7,8,9
PORT=29502 ./tools/dist_train.sh configs/dla/dla102up_80k_new.py 8 --work-dir ./out/dla102up_80k_new_bs16/