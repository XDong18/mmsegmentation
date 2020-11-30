export CUDA_VISIBLE_DEVICES=8,9
PORT=29502 ./tools/dist_train.sh configs/dla/dla34up_10x.py 2 --work-dir ./out/dla34up_10x/