export CUDA_VISIBLE_DEVICES=4,5
python tools/test.py configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_40k_bdd100k.py ./out/deeplabv3plus_r101-d8_769x769_40k_bs16_bdd100k/latest.pth \
--out ./result/deeplabv3plus_r101-d8_769x769_40k_bs16_bdd100k.pkl --show-dir ./show/deeplabv3plus_r101-d8_769x769_40k_bs16_bdd100k