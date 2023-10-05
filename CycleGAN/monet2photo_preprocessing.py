import os

# root path depends on your computer
root = "Y:/user/yrso/data/Horse2zebra/train/b"
save_root = 'Y:/user/yrso/data/Horse2zebra/train/b'
resize_size = 64

img_names = os.listdir(root)

i = 1
for name in img_names:
    before_name = os.path.join(root, name)
    after_name = 'zebra' + str(i) + '.png'
    after_name = os.path.join(root, after_name)
    os.rename(before_name, after_name)
    i += 1

print('complete')