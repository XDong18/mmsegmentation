export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29502 ./tools/dist_train.sh configs/dla/dla34up_10x.py 4 --work-dir ./out/dla34up_10x/