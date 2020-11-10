export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29502 ./tools/dist_train.sh configs/pspnet/pspnet_r50-d8_769x769_40k_bdd100k.py 4 --work-dir /shared/xudongliu/code/f_server/mmsegmentation/out/pspnet_r50-d8_769x769_40k \
--resume-from /shared/xudongliu/code/f_server/mmsegmentation/out/pspnet_r50-d8_769x769_40k/latest.pth