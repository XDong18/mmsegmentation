export CUDA_VISIBLE_DEVICES=4,5
python tools/test.py configs/danet/danet_r101-d8_769x769_40k_bdd100k.py ./out/danet_r101-d8_769x769_40k_bs16_bdd100k/latest.pth \
--out ./result/test/danet_r101-d8_769x769_40k_bdd100k.pkl --show-dir ./show/test/danet_r101-d8_769x769_40k_bdd100k