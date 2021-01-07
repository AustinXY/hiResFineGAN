from PIL import Image
import os


# count = 57000
train_dir = 'images'
depth = 2
res = 32 * 2**depth

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im2.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def generate_imgpath(train_dir, name, res):
    image_dir = '../output/%s' % (train_dir)
    # return '%s/count_%09d_fake_samples%d_res%s.png' % (image_dir, count, i, res)
    return '%s/%s%d.png' % (image_dir, name, res)

im_path = generate_imgpath(train_dir, 'background', res)
im = Image.open(im_path)
for name in ['parent_final', 'child_final', 'child_foreground', 'parent_foreground', 'parent_mask', 'child_mask', 'parent_foreground_masked', 'child_foreground_masked']:
    _im_path = generate_imgpath(train_dir, name, res)
    _im = Image.open(_im_path)
    im = get_concat_v(im, _im)

im.save('../%sevalsample.png' % res)
