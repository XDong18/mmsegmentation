export CUDA_VISIBLE_DEVICES=6,7,8,9
PORT=29502 ./tools/dist_train.sh configs/dla/dla102up_80k_new.py 4 --work-dir ./out/dla102up_80k_new/