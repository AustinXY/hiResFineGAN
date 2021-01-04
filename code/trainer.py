from __future__ import print_function
from six.moves import range
import sys
import numpy as np
import os
import random
import time
from PIL import Image
from copy import deepcopy

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
from torch.nn.functional import softmax, log_softmax
from torch.nn.functional import cosine_similarity
from tensorboardX import summary
from tensorboardX import FileWriter
import torch.nn.functional as F

from miscc.config import cfg
from miscc.utils import mkdir_p

from datasets import Dataset
import torchvision.transforms as transforms
from datetime import datetime


from model import G_NET, D_NET_PC


start_depth = cfg.TRAIN.START_DEPTH
end_depth = cfg.TRAIN.END_DEPTH
batchsize_per_depth = cfg.TRAIN.BATCHSIZE_PER_DEPTH
blend_epochs_per_depth = cfg.TRAIN.BLEND_EPOCHS_PER_DEPTH
stable_epochs_per_depth = cfg.TRAIN.STABLE_EPOCHS_PER_DEPTH

# ################## Shared functions ###################

def child_to_parent(child_c_code, classes_child, classes_parent):
    ratio = classes_child / classes_parent
    arg_parent = torch.argmax(child_c_code,  dim = 1) / ratio
    parent_c_code = torch.zeros([child_c_code.size(0), classes_parent]).cuda()
    for i in range(child_c_code.size(0)):
        parent_c_code[i][arg_parent[i].type(torch.LongTensor)] = 1
    return parent_c_code


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, 1.0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname == 'MnetConv':
        nn.init.constant_(m.mask_conv.weight.data, 1.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_network(gpus):
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    # print(netG)

    netsD = []
    netsD.append(D_NET_PC(0))
    netsD.append(D_NET_PC(1))
    netsD.append(D_NET_PC(2))

    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
        # print(netsD[i])

    count = 0

    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('netG_') + 5
        iend = cfg.TRAIN.NET_G.rfind('_depth')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count)
        istart = cfg.TRAIN.NET_G.rfind('depth')
        iend = cfg.TRAIN.NET_G.rfind('.')
        _depth = cfg.TRAIN.NET_G[istart:iend]

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s%d_%s.pth' % (cfg.TRAIN.NET_D, i, _depth))
            state_dict = torch.load('%s%d_%s.pth' % (cfg.TRAIN.NET_D, i, _depth))
            netsD[i].load_state_dict(state_dict)

    if cfg.CUDA:
        netG.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()

    return netG, netsD, len(netsD), count


def define_optimizers(netG, netsD):
    optimizersD = []
    # num_Ds = len(netsD)
    # for i in range(num_Ds):
    opt = optim.Adam(netsD[2].parameters(),
                        lr=cfg.TRAIN.DISCRIMINATOR_LR,
                        betas=(0.5, 0.999))
    optimizersD.append(opt)

    optimizerG = []
    optimizerG.append(optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999)))

    opt = optim.Adam(netsD[0].parameters(),
                     lr=cfg.TRAIN.GENERATOR_LR,
                     betas=(0.5, 0.999))
    optimizerG.append(opt)

    opt = optim.Adam(netsD[1].parameters(),
                     lr=cfg.TRAIN.GENERATOR_LR,
                     betas=(0.5, 0.999))
    optimizerG.append(opt)

    opt = optim.Adam([{'params': netsD[2].module.down_net[0].jointConv.parameters()},
                      {'params': netsD[2].module.down_net[0].logits.parameters()}],
                     lr=cfg.TRAIN.GENERATOR_LR,
                     betas=(0.5, 0.999))
    optimizerG.append(opt)

    return optimizerG, optimizersD


def save_model(netG, avg_param_G, netsD, epoch, model_dir, cur_depth):
    load_params(netG, avg_param_G)
    torch.save(
        netG.state_dict(),
        '%s/netG_%d_depth%d.pth' % (model_dir, epoch, cur_depth))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(netD.state_dict(),
            '%s/netD%d_depth%d.pth' % (model_dir, i, cur_depth))
    print('Save G/Ds models.')


def save_img_results(fake_imgs, count, image_dir, summary_writer, depth):
    num = cfg.TRAIN.VIS_COUNT
    for i in range(len(fake_imgs)):
        fake_img = fake_imgs[i][0:num]

        vutils.save_image(
            fake_img.data, '%s/count_%09d_fake_samples%d_depth%d.png' %
            (image_dir, count, i, depth), normalize=True)

        fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

        fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
        fake_img_set = (fake_img_set + 1) * 255 / 2
        fake_img_set = fake_img_set.astype(np.uint8)
        summary_writer.flush()
    print('Save image samples.')

class FineGAN_trainer(object):
    def __init__(self, output_dir):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True


    def prepare_data(self, data):
        cimgs, c_code, _ = data
        if cfg.CUDA:
            vc_code = Variable(c_code).cuda()
            real_vcimgs = Variable(cimgs).cuda()
        else:
            vc_code = Variable(c_code)
            real_vcimgs = Variable(cimgs)
        return real_vcimgs, vc_code


    def train_Dnet(self, idx, count):
        flag = count % 100
        criterion, criterion_one = self.criterion, self.criterion_one

        netD, optD = self.netsD[idx], self.optimizersD[0]

        real_imgs = self.real_cimgs
        masks = None

        fake_imgs = self.fake_imgs[idx]
        netD.zero_grad()
        real_logits = netD(real_imgs, self.alpha, masks)

        fake_labels = torch.zeros_like(real_logits[1])
        real_labels = torch.ones_like(real_logits[1])

        fake_logits = netD(fake_imgs.detach(), self.alpha)

        errD_real = criterion_one(real_logits[1], real_labels) # Real/Fake loss for the real image
        errD_fake = criterion_one(fake_logits[1], fake_labels) # Real/Fake loss for the fake image
        errD = errD_real + errD_fake

        errD.backward()
        optD.step()

        if flag == 0:
            summary_D = summary.scalar('D_loss%d' % idx, errD.item())
            self.summary_writer.add_summary(summary_D, count)
            summary_D_real = summary.scalar('D_loss_real_%d' % idx, errD_real.item())
            self.summary_writer.add_summary(summary_D_real, count)
            summary_D_fake = summary.scalar('D_loss_fake_%d' % idx, errD_fake.item())
            self.summary_writer.add_summary(summary_D_fake, count)

        return errD

    def train_Gnet(self, count):
        self.netG.zero_grad()
        for myit in range(len(self.netsD)):
             self.netsD[myit].zero_grad()

        errG_total = 0
        flag = count % 100
        criterion_one, criterion_class, c_code, p_code, b_code = self.criterion_one, self.criterion_class, self.c_code, self.p_code, self.b_code
        fg_mk = self.mk_imgs[0]
        bg_img = self.fake_imgs[0]
        bg_mk = torch.ones_like(fg_mk) - fg_mk
        bg_of_bg = bg_mk * bg_img

        p_info_wt = 1.
        c_info_wt = 1.
        b_info_wt = 1.
        for i in range(self.num_Ds):
            if i == 2:  # real/fake loss for background (0) and child (2) stage
                outputs = self.netsD[i](self.fake_imgs[i], self.alpha)
                real_labels = torch.ones_like(outputs[1])
                errG = criterion_one(outputs[1], real_labels)
                errG_total = errG_total + errG

            if i == 1: # Mutual information loss for the parent stage (1)
                pred_p = self.netsD[i](self.fg_mk[0], self.alpha)
                errG_info = criterion_class(pred_p[0], torch.nonzero(p_code.long())[:,1]) * p_info_wt
            elif i == 2: # Mutual information loss for the child stage (2)
                pred_c = self.netsD[i](self.fg_mk[1], self.alpha)
                errG_info = criterion_class(pred_c[0], torch.nonzero(c_code.long())[:,1]) * c_info_wt
            else: # Mutual information loss for the bg stage (0)
                pred_b = self.netsD[i](bg_img, self.alpha)
                errG_info = criterion_class(pred_b[0], torch.nonzero(b_code.long())[:,1]) * b_info_wt

            errG_total = errG_total + errG_info

            if flag == 0:
                summary_D_class = summary.scalar('Information_loss_%d' % i, errG_info.item())
                self.summary_writer.add_summary(summary_D_class, count)

                if i == 2:
                  summary_D = summary.scalar('G_loss%d' % i, errG.item())
                  self.summary_writer.add_summary(summary_D, count)

        # fg_mk = self.mk_imgs[0]
        # bg_mk = torch.ones_like(fg_mk) - fg_mk
        ch_mk = self.mk_imgs[1]
        ms = fg_mk.size()
        min_fg_cvg = cfg.TRAIN.MIN_FG_CVG * ms[2] * ms[3]
        # min_bg_cvg = cfg.TRAIN.MIN_BG_CVG * ms[2] * ms[3]
        binary_loss = self.binarization_loss(fg_mk) * 10
        oob_loss = torch.sum(bg_mk * ch_mk, dim=(-1,-2)).mean() * 1e-2
        # oob_loss = torch.sum(bg_mk * ch_mk, dim=(-1,-2)).mean() * 0
        fg_cvg_loss = F.relu(min_fg_cvg - torch.sum(fg_mk, dim=(-1,-2))).mean() * 1e-2
        # bg_cvg_loss = F.relu(min_bg_cvg - torch.sum(bg_mk, dim=(-1,-2))).mean() * 0

        errG_total += binary_loss + fg_cvg_loss + oob_loss  # + bg_cvg_loss + oob_loss

        self.cl = fg_cvg_loss # + bg_cvg_loss
        self.bl = binary_loss
        self.ol = oob_loss

        errG_total.backward()
        for myit in range(3):
            self.optimizerG[myit].step()
        return errG_total

    def concentration_loss(self, fg_mk, bg_mk):
        eps = 1e-12
        fg_mass = torch.sum(fg_mk, dim=(-1, -2)) + eps
        bg_mass = torch.sum(bg_mk, dim=(-1, -2)) + eps
        center_x = torch.sum(fg_mk * self.xc, dim=(-1,-2)) / fg_mass
        center_y = torch.sum(fg_mk * self.yc, dim=(-1,-2)) / fg_mass
        center_x = center_x.unsqueeze(2).unsqueeze(3)
        center_y = center_y.unsqueeze(2).unsqueeze(3)
        fg_dist = (self.xc - center_x * torch.ones_like(self.xc))**2 + \
            (self.yc - center_y * torch.ones_like(self.yc))**2
        bg_dist = (self.xc - center_x * torch.ones_like(self.xc))**2 + \
            (self.yc - center_y * torch.ones_like(self.yc))**2
        fg_var = torch.sum(fg_dist * fg_mk, dim=(-1, -2)) / fg_mass
        bg_var = torch.sum(bg_dist * bg_mk, dim=(-1, -2)) / bg_mass
        return F.relu(fg_var - bg_var).mean()

    def get_dataloader(self, cur_depth):
        bshuffle = True
        imsize = 32 * (2 ** cur_depth)
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])

        dataset = Dataset(cfg.DATA_DIR,
                          cur_depth=cur_depth,
                          transform=image_transform)
        assert dataset
        num_gpu = len(cfg.GPU_ID.split(','))
        bs = batchsize_per_depth[cur_depth]
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size= bs * num_gpu,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

        imsize = 32 * (2 ** cur_depth)
        xc, yc = torch.meshgrid([torch.arange(imsize), torch.arange(imsize)])
        self.xc = xc.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1).float().cuda()
        self.yc = yc.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1).float().cuda()
        return dataloader

    def train(self):
        self.netG, self.netsD, self.num_Ds, start_count = load_network(self.gpus)
        newly_loaded = True
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.BCELoss(reduce=False)
        self.criterion_one = nn.BCELoss()
        self.criterion_class = nn.CrossEntropyLoss()

        nz = cfg.GAN.Z_DIM

        if cfg.CUDA:
            self.criterion.cuda()
            self.criterion_one.cuda()
            self.criterion_class.cuda()

        print ("Starting normal FineGAN training..")
        count = start_count

        for cur_depth in range(start_depth, end_depth+1):
            max_epoch = blend_epochs_per_depth[cur_depth] + \
                stable_epochs_per_depth[cur_depth]
            dataloader = self.get_dataloader(cur_depth)
            num_batches = len(dataloader)

            depth_ep_ctr = 0  # depth epoch counter
            batch_size = batchsize_per_depth[cur_depth] * self.num_gpus

            noise = Variable(torch.FloatTensor(batch_size, nz))
            fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))

            if cfg.CUDA:
                noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

            start_epoch = start_count // (num_batches)
            start_count = 0
            self.beta = 0

            for epoch in range(start_epoch, max_epoch):
                depth_ep_ctr += 1

                # switch dataset
                if depth_ep_ctr < blend_epochs_per_depth[cur_depth]:
                    self.alpha = depth_ep_ctr / blend_epochs_per_depth[cur_depth]
                else:
                    self.alpha = 1

                start_t = time.time()
                for step, data in enumerate(dataloader, 0):

                    # _travel = 5000.0
                    # if count < _travel:
                    #     self.beta = count / _travel
                    # else:
                    #     self.beta = 1
                    self.beta = 1

                    count += 1
                    self.real_cimgs, self.c_code = self.prepare_data(data)

                    # Feedforward through Generator. Obtain stagewise fake images
                    noise.data.normal_(0, 1)
                    self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk, pb_code = self.netG(noise, self.c_code, self.alpha, self.beta)

                    self.p_code, self.b_code = pb_code

                    # Obtain the parent code given the child code
                    # self.p_code = child_to_parent(self.c_code, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES)

                    # Update Discriminator networks
                    errD_total = self.train_Dnet(2, count)

                    # Update the Generator networks
                    errG_total = self.train_Gnet(count)
                    for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                        avg_p.mul_(0.999).add_(0.001, p.data)

                    newly_loaded = False
                    if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                        print("binary_loss: {}, cvg_loss: {}, oob_loss: {}".
                              format(self.bl.item(), self.cl.item(), self.ol.item()))

                        backup_para = copy_G_params(self.netG)
                        if count % cfg.TRAIN.SAVEMODEL_INTERVAL == 0:
                            save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir, cur_depth)
                        # Save images
                        load_params(self.netG, avg_param_G)

                        fake_imgs, fg_imgs, mk_imgs, fg_mk, _ = self.netG(fixed_noise, self.c_code, self.alpha)
                        save_img_results((fake_imgs + fg_imgs + mk_imgs + fg_mk),
                                         count, self.image_dir, self.summary_writer, cur_depth)
                        #
                        load_params(self.netG, backup_para)

                end_t = time.time()
                print('''[%d/%d][%d]Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                      % (epoch, max_epoch, num_batches,
                        errD_total.item(), errG_total.item(),
                        end_t - start_t))

            if not newly_loaded:
                save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir, cur_depth)
            self.update_network()
            avg_param_G = copy_G_params(self.netG)


    def binarization_loss(self, mask):
        return torch.min(1-mask, mask).mean()


    def update_network(self):
        self.netG.module.inc_depth()
        # self.netG = torch.nn.DataParallel(self.netG, device_ids=self.gpus)
        # print(self.netG)

        for netD in self.netsD:
            netD.module.inc_depth()
            # netD = torch.nn.DataParallel(netD, device_ids=self.gpus)
            # print(netD)

        if cfg.CUDA:
            self.netG.cuda()
            for netD in self.netsD:
                netD.cuda()

        self.optimizersD = []
        for netD in self.netsD:
            opt = optim.Adam(netD.parameters(),
                lr=cfg.TRAIN.DISCRIMINATOR_LR,
                betas=(0.5, 0.999))
            self.optimizersD.append(opt)

        self.optimizerG = []
        self.optimizerG.append(optim.Adam(self.netG.parameters(),
            lr=cfg.TRAIN.GENERATOR_LR,
            betas=(0.5, 0.999)))

        opt = optim.Adam(self.netsD[1].parameters(),
                        lr=cfg.TRAIN.GENERATOR_LR,
                        betas=(0.5, 0.999))
        self.optimizerG.append(opt)

        opt = optim.Adam([{'params': self.netsD[2].module.down_net[0].jointConv.parameters()},
                        {'params': self.netsD[2].module.down_net[0].logits.parameters()}],
                        lr=cfg.TRAIN.GENERATOR_LR,
                        betas=(0.5, 0.999))
        self.optimizerG.append(opt)


class FineGAN_evaluator(object):

    def __init__(self):

        self.save_dir = os.path.join(cfg.SAVE_DIR, 'images')
        mkdir_p(self.save_dir)
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus

    def evaluate_finegan(self):
        random.seed(datetime.now())
        # torch.manual_seed(random.randint(0, 9999))
        torch.manual_seed(2)

        depth = cfg.TRAIN.START_DEPTH
        res = 32 * 2 ** depth
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for model not found!')
        else:
            # Build and load the generator
            netG = G_NET()
            # print(netG)
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            model_dict = netG.state_dict()

            state_dict = \
                torch.load(cfg.TRAIN.NET_G,
                           map_location=lambda storage, loc: storage)

            state_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict}

            model_dict.update(state_dict)
            netG.load_state_dict(model_dict)
            print('Load ', cfg.TRAIN.NET_G)

            # Uncomment this to print Generator layers
            # print(netG)

            nz = cfg.GAN.Z_DIM
            noise = torch.FloatTensor(1, nz)

            noise.data.normal_(0, 1)
            noise = noise.repeat(self.batch_size, 1)

            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            netG.eval()
            bg_categories = cfg.FINE_GRAINED_CATEGORIES // cfg.NUM_C_PER_B

            b = random.randint(0, bg_categories-1)
            p = random.randint(0, cfg.SUPER_CATEGORIES-1)
            c = random.randint(0, cfg.FINE_GRAINED_CATEGORIES-1)
            bg_code = torch.zeros([self.batch_size, bg_categories])
            p_code = torch.zeros([self.batch_size, cfg.SUPER_CATEGORIES])
            c_code = torch.zeros([self.batch_size, cfg.FINE_GRAINED_CATEGORIES])

            bg_li = []
            pf_li = []
            cf_li = []
            pk_li = []
            ck_li = []
            pfg_li = []
            cfg_li = []
            pfgmk_li = []
            cfgmk_li = []
            b_li = np.random.permutation(bg_categories-1)
            p_li = np.random.permutation(cfg.SUPER_CATEGORIES-1)
            c_li = np.random.permutation(cfg.FINE_GRAINED_CATEGORIES-1)

            # b_li = np.array(range(0, bg_categories))
            # p_c_dict = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            #           1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            #           2: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            #           3: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            #           4: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            #           5: [50, 51, 52, 53, 54, 55, 56, 57, 58],
            #           6: [59, 60, 61, 62, 63, 64, 65, 66, 67, 68],
            #           7: [69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
            #           8: [79, 80, 81, 82, 83, 84, 85, 86, 87, 88],
            #           9: [89, 90, 91, 92, 93, 94, 95, 96, 97, 98],
            #           10: [99, 100, 101, 102, 103, 104, 105, 106, 107],
            #           11: [108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
            #           12: [118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
            #           13: [128, 129, 130, 131, 132, 133, 134, 135, 136, 137],
            #           14: [138, 139, 140, 141, 142, 143, 144, 145, 146, 147],
            #           15: [148, 149, 150, 151, 152, 153, 154, 155, 156],
            #           16: [157, 158, 159, 160, 161, 162, 163, 164, 165, 166],
            #           17: [167, 168, 169, 170, 171, 172, 173, 174, 175, 176],
            #           18: [177, 178, 179, 180, 181, 182, 183, 184, 185, 186],
            #           19: [187, 188, 189, 190, 191, 192, 193, 194, 195]}

            # c_li = p_c_dict[19]
            # c_li = np.array(range(0, 98))
            c_li = np.array(range(98, 196))
            nrow = 10
            for k in range(1):
                b = b_li[k]
                p = p_li[k]
                c = c_li[k]

                for i in range(len(c_li)):
                    bg_code = torch.zeros([self.batch_size, bg_categories])
                    p_code = torch.zeros([self.batch_size, cfg.SUPER_CATEGORIES])
                    c_code = torch.zeros([self.batch_size, cfg.FINE_GRAINED_CATEGORIES])

                    # noise.data.normal_(0, 1)
                    # b = b_li[i]
                    # p = p_li[i]
                    c = c_li[i]
                    b = c % bg_categories
                    p = int(c // 9.8)
                    # print('b:', b, 'p:', p, 'c:', c)
                    # p = i
                    for j in range(self.batch_size):
                        bg_code[j][b] = 1
                        p_code[j][p] = 1
                        c_code[j][c] = 1

                    fake_imgs, fg_imgs, mk_imgs, fgmk_imgs, _ = netG(
                        noise, c_code, None, p_code, bg_code)  # Forward pass through the generator
                    bg_li.append(fake_imgs[0][0])
                    pf_li.append(fake_imgs[1][0])
                    cf_li.append(fake_imgs[2][0])
                    pk_li.append(mk_imgs[0][0])
                    ck_li.append(mk_imgs[1][0])
                    pfg_li.append(fg_imgs[0][0])
                    cfg_li.append(fg_imgs[1][0])
                    pfgmk_li.append(fgmk_imgs[0][0])
                    cfgmk_li.append(fgmk_imgs[1][0])

            save_image(bg_li, self.save_dir, 'background', nrow, res)
            save_image(pf_li, self.save_dir, 'parent_final', nrow, res)
            save_image(cf_li, self.save_dir, 'child_final', nrow, res)
            save_image(pfg_li, self.save_dir, 'parent_foreground', nrow, res)
            save_image(cfg_li, self.save_dir, 'child_foreground', nrow, res)
            save_image(pk_li, self.save_dir, 'parent_mask', nrow, res)
            save_image(ck_li, self.save_dir, 'child_mask', nrow, res)
            save_image(pfgmk_li, self.save_dir,
                       'parent_foreground_masked', nrow, res)
            save_image(cfgmk_li, self.save_dir,
                       'child_foreground_masked', nrow, res)

    def save_image(self, images, save_dir, iname):

        img_name = '%s.png' % (iname)
        full_path = os.path.join(save_dir, img_name)

        if (iname.find('mask') == -1) or (iname.find('foreground') != -1):
            img = images.add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(full_path)

        else:
            img = images.mul(255).clamp(0, 255).byte()
            ndarr = img.data.cpu().numpy()
            ndarr = np.reshape(ndarr, (ndarr.shape[-1], ndarr.shape[-1], 1))
            ndarr = np.repeat(ndarr, 3, axis=2)
            im = Image.fromarray(ndarr)
            im.save(full_path)


def save_image(fake_imgs, image_dir, iname, nrow, res):
    img_name = '%s%d.png' % (iname, res)
    vutils.save_image(fake_imgs, '%s/%s' %
                      (image_dir, img_name), nrow=nrow, normalize=True)
