import os

img_dir = 'show/deeplabv3plus_r50-d8_769x769_40k_torchvision'
img_list = sorted(os.listdir(img_dir))
out_fn = 'show/deeplabv3plus_r50-d8_769x769_40k_torchvision_val.html'
with open (out_fn, 'w') as f:
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<h1>deeplabv3plus_r50-d8_769x769_40k_torchvision_val</h1>\n')

    for i, img in enumerate(img_list):
        name = "<p>" + str(i) + ' ---- ' + img + "</p>\n"
        f.write(name)
        new_line = "<img src='" + os.path.join('deeplabv3plus_r50-d8_769x769_40k_torchvision', img) + "'>\n"
        f.write(new_line)

    # f.write("<img src='" + "test.jpg" + "'>\n")
    f.write('</body>\n')
    f.write('</html>\n')
