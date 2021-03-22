export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29507 ./tools/dist_train.sh \
configs/dla_lane/dla34up_lane_12e.py 4 \
--work-dir ./out/dla34up_lane_12e_iter