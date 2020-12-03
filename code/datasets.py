from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch
from copy import deepcopy
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, cur_depth, bbox=None,
             transform=None, normalize=None):
    imsize = 32 * (2 ** cur_depth)
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        fimg = deepcopy(img)
        fimg_arr = np.array(fimg)
        fimg = Image.fromarray(fimg_arr)
        cimg = img.crop([x1, y1, x2, y2])

    if transform is not None:
        cimg = transform(cimg)

    retc = []
    re_cimg = transforms.Scale(imsize)(cimg)

    retc = normalize(re_cimg)
    return retc


class Dataset(data.Dataset):
    def __init__(self, data_dir, cur_depth, transform=None):

        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.cur_depth = cur_depth
        # self.imsize = 32 * (2 ** cur_depth)

        self.data = []
        self.data_dir = data_dir
        self.bbox = self.load_bbox()
        self.filenames = self.load_filenames(data_dir)
        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    # only used in background stage

    def load_bbox(self):
        # Returns a dictionary with image filename as 'key' and its bounding box coordinates as 'value'

        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        filenames = [fname[:-4] for fname in filenames]
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        data_dir = self.data_dir
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        cimgs = get_imgs(img_name, self.cur_depth,
                                       bbox, self.transform, normalize=self.norm)

        # Randomly generating child code during training
        rand_class = random.sample(range(cfg.FINE_GRAINED_CATEGORIES), 1)
        c_code = torch.zeros([cfg.FINE_GRAINED_CATEGORIES, ])
        c_code[rand_class] = 1
        return cimgs, c_code, key

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        data_dir = self.data_dir
        c_code = self.c_code[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.cur_depth,
                              bbox, self.transform, normalize=self.norm)

        return imgs, c_code, key

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)
