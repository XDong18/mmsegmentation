import os

img_dir = 'show/dla34up_80k_new_sbn'
img_list = sorted(os.listdir(img_dir))
out_fn = 'show/dla34up_80k_new_sbn.html'
with open (out_fn, 'w') as f:
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<h1>dla34up_80k_new_sbn</h1>\n')

    for i, img in enumerate(img_list):
        name = "<p>" + str(i) + ' ---- ' + img + "</p>\n"
        f.write(name)
        new_line = "<img src='" + os.path.join('dla34up_80k_new_sbn', img) + "'>\n"
        f.write(new_line)

    # f.write("<img src='" + "test.jpg" + "'>\n")
    f.write('</body>\n')
    f.write('</html>\n')
