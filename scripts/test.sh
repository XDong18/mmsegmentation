export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29503 ./tools/dist_test.sh \
configs/dla_lane/dla34up_lane_12e.py \
out/dla34up_lane_12e_iter/latest.pth 4