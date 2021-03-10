export CUDA_VISIBLE_DEVICES=4,5

python tools/test.py configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_40k_bdd100k.py ./out/deeplabv3plus_r101-d8_769x769_40k_bs16_bdd100k/latest.pth \
--out ./result/test/deeplabv3plus_r101-d8_769x769_40k_bdd100k.pkl --show-dir ./show/test/deeplabv3plus_r101-d8_769x769_40k_bdd100k

python tools/test.py configs/danet/danet_r101-d8_769x769_40k_bdd100k.py ./out/danet_r101-d8_769x769_40k_bs16_bdd100k/latest.pth \
--out ./result/test/danet_r101-d8_769x769_40k_bdd100k.pkl --show-dir ./show/test/danet_r101-d8_769x769_40k_bdd100k

python tools/test.py configs/emanet/emanet_r101-d8_769x769_40k_bdd100k.py ./out/emanet_r101-d8_769x769_40k_bdd100k/latest.pth \
--out ./result/test/emanet_r101-d8_769x769_40k_bdd100k.pkl --show-dir ./show/test/emanet_r101-d8_769x769_40k_bdd100k

python tools/test.py configs/pspnet/pspnet_r101-d8_769x769_40k_bdd100k.py ./out/pspnet_r101-d8_769x769_40k_bs16_bdd100k/latest.pth \
--out ./result/test/pspnet_r101-d8_769x769_40k_bdd100k.pkl --show-dir ./show/test/pspnet_r101-d8_769x769_40k_bdd100k