from PIL import Image
import os


eval_dir = '../output/images/'


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im2.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


im_li = []
for im_path in os.listdir(eval_dir):
    im_li.append(eval_dir + im_path)

im_li.sort()

im = Image.open(im_li[0])
for i in range(1, 10):
    _im = Image.open(im_li[i])
    im = get_concat_v(im, _im)

im.save('../eval.png')
