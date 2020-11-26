from PIL import Image
import os


count = 98000
train_dir = 'CUB_2020_11_22_07_20_18'


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im2.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def generate_imgpath(train_dir, count, i, depth):
    image_dir = '../output/%s/Image' % (train_dir)
    # return '%s/count_%09d_fake_samples%d_res%s.png' % (image_dir, count, i, res)
    return '%s/count_%09d_fake_samples%d_depth%d.png' % (image_dir, count, i, depth)


for im_path in os.listdir('../output/%s/Image/' % train_dir):
    if ('%09d' % count) in im_path:
        # im_path_li.append(im_path)
        # istart = im_path.rfind('res') + 3
        # iend = im_path.rfind('.png')
        # res = im_path[istart:iend]
        depth = 2
        break


im_path = generate_imgpath(train_dir, count, 0, depth)
im = Image.open(im_path)
for i in range(1, 9):
    _im_path = generate_imgpath(train_dir, count, i, depth)
    _im = Image.open(_im_path)
    im = get_concat_v(im, _im)

res = 32 * 2**depth
im.save('../%ssample.png' % res)
