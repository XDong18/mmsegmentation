export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PORT=29507 ./tools/dist_train.sh \
configs/pspnet/pspnet_r101-d8_769x769_40k_bdd100k.py 8 \
--work-dir ./out/pspnet_r101-d8_769x769_40k_bs16_bdd100k/