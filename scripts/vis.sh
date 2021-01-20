export CUDA_VISIBLE_DEVICES=6,7,8,9
python tools/test.py configs/dla/dla102up_80k_new.py ./out/dla_model/dla102_bs8_500e_new_pretrained.pth --eval mIoU