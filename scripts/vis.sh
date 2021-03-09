export CUDA_VISIBLE_DEVICES=4,5
python tools/test.py configs/emanet/emanet_r101-d8_769x769_40k_bdd100k.py./out/emanet_r101-d8_769x769_40k_bdd100k/latest.pth \
--out ./result/test/emanet_r101-d8_769x769_40k_bdd100k.pkl --show-dir ./show/test/emanet_r101-d8_769x769_40k_bdd100k
