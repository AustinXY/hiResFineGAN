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
import json


from model import G_NET, D_NET_PC, Bi_Dis, MaskPredictor


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
    netsD.append(D_NET_PC(0))  # bg info
    netsD.append(D_NET_PC(1))  # fg info

    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)

    bi_netD = Bi_Dis()
    bi_netD.apply(weights_init)
    bi_netD = torch.nn.DataParallel(bi_netD, device_ids=gpus)

    mkpred_net = MaskPredictor(3)
    mkpred_net.apply(weights_init)
    mkpred_net = torch.nn.DataParallel(mkpred_net, device_ids=gpus)

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

    return netG, netsD, bi_netD, mkpred_net, count


def define_optimizers(netG, netsD, bi_netD, mkpred_net):
    optimizersD = []
    opt = optim.Adam(netsD[1].parameters(),
                     lr=cfg.TRAIN.DISCRIMINATOR_LR,
                     betas=(0.5, 0.999))
    optimizersD.append(opt)

    optimizersBiD = optim.Adam(bi_netD.parameters(),
                     lr=cfg.TRAIN.DISCRIMINATOR_LR,
                     betas=(0.5, 0.999))

    optimizersMk = optim.Adam(mkpred_net.parameters(),
                              lr=cfg.TRAIN.DISCRIMINATOR_LR,
                              betas=(0.5, 0.999))

    optimizerG = []
    opt = optim.Adam(netG.parameters(),
                     lr=cfg.TRAIN.GENERATOR_LR,
                     betas=(0.5, 0.999))
    optimizerG.append(opt)

    opt = optim.Adam(netsD[0].parameters(),
                     lr=cfg.TRAIN.GENERATOR_LR,
                     betas=(0.5, 0.999))
    optimizerG.append(opt)

    opt = optim.Adam([{'params': netsD[1].module.down_net[0].jointConv.parameters()},
                      {'params': netsD[1].module.down_net[0].logits.parameters()}],
                     lr=cfg.TRAIN.GENERATOR_LR,
                     betas=(0.5, 0.999))
    optimizerG.append(opt)

    return optimizerG, optimizersD, optimizersBiD, optimizersMk


def save_model(netG, avg_param_G, netsD, bi_netD, mkpred_net, mapping, epoch, model_dir, cur_depth):
    load_params(netG, avg_param_G)
    torch.save(netG.state_dict(),
               '%s/netG_%d_depth%d.pth' % (model_dir, epoch, cur_depth))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(netD.state_dict(),
                   '%s/netD%d_%d_depth%d.pth' % (model_dir, i, epoch, cur_depth))

    torch.save(bi_netD.state_dict(),
               '%s/binetD_%d_depth%d.pth' % (model_dir, epoch, cur_depth))

    torch.save(mkpred_net.state_dict(),
               '%s/mkprednet_%d_depth%d.pth' % (model_dir, epoch, cur_depth))

    with open('%s/mapping_%d.txt' % (model_dir, epoch), "w") as f:
        json.dump(mapping, f)

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
        cimgs, _ = data
        if cfg.CUDA:
            real_vcimgs = Variable(cimgs).cuda()
        else:
            real_vcimgs = Variable(cimgs)
        return real_vcimgs


    def train_Dnet(self, count):
        netD, optD = self.netsD[1], self.optimizersD[0]
        netD.zero_grad()

        flag = count % 100
        criterion_one = self.criterion_one

        real_imgs = self.real_cimgs
        fake_imgs = self.fake_img[0].detach()

        real_logits = netD(real_imgs)[1]
        fake_logits = netD(fake_imgs)[1]

        fake_labels = torch.zeros_like(real_logits)
        real_labels = torch.ones_like(fake_logits)


        errD_real = criterion_one(real_logits, real_labels) # Real/Fake loss for the real image
        errD_fake = criterion_one(fake_logits, fake_labels) # Real/Fake loss for the fake image
        errD = errD_real + errD_fake

        errD.backward()
        optD.step()

        if flag == 0:
            summary_D = summary.scalar('D_loss', errD.item())
            self.summary_writer.add_summary(summary_D, count)
            summary_D_real = summary.scalar('D_loss_real_1', errD_real.item())
            self.summary_writer.add_summary(summary_D_real, count)
            summary_D_fake = summary.scalar('D_loss_fake_1', errD_fake.item())
            self.summary_writer.add_summary(summary_D_fake, count)

        return errD


    # keep length of pb pair stored not changing
    def real_pb_pair_log(self, pid, bid):
        popped_b = self.real_pb_pair[pid].pop(0)
        self.real_pb_pair[pid].append(bid)
        self.mapped_b_count[popped_b] -= 1
        self.mapped_b_count[bid] += 1

        if self.mapped_b_count[popped_b] < self.overused_thld:
            self.not_overused_b.add(popped_b)
            if self.mapped_b_count[popped_b] < self.underuse_thld:
                self.underused_b.add(popped_b)

        if (bid in self.not_overused_b) and \
           (self.mapped_b_count[bid] >= self.overused_thld):
            self.not_overused_b.remove(bid)

        if (bid in self.underused_b) and \
           (self.mapped_b_count[bid] >= self.underuse_thld):
            self.underused_b.remove(bid)


    def train_BiDnet(self, count):
        optBiD = self.optimizersBiD
        self.bi_netD.zero_grad()
        self.mkpred_net.zero_grad()

        flag = count % 100
        criterion_one = self.criterion_one

        real_imgs = self.real_cimgs
        fake_imgs = self.fake_img[0].detach()

        with torch.no_grad():
            real_pred_mk = self.mkpred_net(real_imgs)
            # fake_pred_mk = self.mkpred_net(fake_imgs)

        real_bg = (torch.ones_like(real_pred_mk) - real_pred_mk) * real_imgs
        real_fg = real_pred_mk * real_imgs

        # fake_bg = (torch.ones_like(fake_pred_mk) - fake_pred_mk) * fake_imgs
        # fake_fg = fake_pred_mk * fake_imgs

        with torch.no_grad():
            real_pred_b = self.netsD[0](real_bg)[0]
            real_pred_p = self.netsD[1](real_fg)[0]

            # fake_pred_b = self.netsD[0](fake_bg)[0]
            # fake_pred_p = self.netsD[1](fake_fg)[0]

            real_bid = torch.argmax(real_pred_b, dim=1)
            real_pid = torch.argmax(real_pred_p, dim=1)
            real_pred_b = torch.zeros_like(self.b_code)
            real_pred_p = torch.zeros_like(self.p_code)
            for i in range(real_bid.size(0)):
                bid = real_bid[i]
                pid = real_pid[i]
                real_pred_b[i, bid] = 1
                real_pred_p[i, pid] = 1

                self.real_pb_pair_log(pid.item(), bid.item())

            fake_pred_b = self.b_code
            fake_pred_p = self.p_code

        real_logits_b, real_logits_p = self.bi_netD(real_imgs, real_pred_b, real_pred_p, real_pred_mk)
        fake_logits_b, fake_logits_p = self.bi_netD(fake_imgs, fake_pred_b, fake_pred_p, self.mk_img[0].detach())

        real_labels = torch.ones_like(real_logits_b)
        fake_labels = torch.zeros_like(fake_logits_b)

        # Real/Fake loss for the real_img+b pair and real_img+p pair
        errBiD_real_b = criterion_one(real_logits_b, real_labels)
        errBiD_real_p = criterion_one(real_logits_p, real_labels)

        # Real/Fake loss for the fake_img+b pair and fake_img+p pair
        errBiD_fake_b = criterion_one(fake_logits_b, fake_labels)
        errBiD_fake_p = criterion_one(fake_logits_p, fake_labels)

        errBiD = errBiD_real_b + errBiD_real_p + errBiD_fake_b + errBiD_fake_p

        errBiD.backward()
        optBiD.step()

        if flag == 0:
            summary_BiD = summary.scalar('BiD_loss', errBiD.item())
            self.summary_writer.add_summary(summary_BiD, count)
            summary_BiD_real = summary.scalar('BiD_loss_real_0', errBiD_real_b.item())
            self.summary_writer.add_summary(summary_BiD_real, count)
            summary_BiD_real = summary.scalar('BiD_loss_real_1', errBiD_real_p.item())
            self.summary_writer.add_summary(summary_BiD_real, count)
            summary_BiD_fake = summary.scalar('BiD_loss_fake_0', errBiD_fake_b.item())
            self.summary_writer.add_summary(summary_BiD_fake, count)
            summary_BiD_fake = summary.scalar('BiD_loss_fake_1', errBiD_fake_p.item())
            self.summary_writer.add_summary(summary_BiD_fake, count)

        return errBiD


    def train_mkpred_net(self, count):
        optMk = self.optimizersMk
        self.mkpred_net.zero_grad()
        criterion = nn.MSELoss()

        flag = count % 100

        fake_imgs = self.fake_img[0].detach()

        pred_mk = self.mkpred_net(fake_imgs)
        real_mk = self.mk_img[0].detach()

        mk_recon_loss = criterion(pred_mk, real_mk)
        binary_loss = self.binarization_loss(pred_mk) * 1

        errMk = mk_recon_loss + binary_loss

        errMk.backward()
        optMk.step()

        if flag == 0:
            summary_mkrecon = summary.scalar('mk_recon_loss', mk_recon_loss.item())
            self.summary_writer.add_summary(summary_mkrecon, count)

        return mk_recon_loss


    def train_Gnet(self, count):
        self.netG.zero_grad()
        self.mkpred_net.zero_grad()
        self.bi_netD.zero_grad()
        for netD in self.netsD:
            netD.zero_grad()

        errG_total = 0
        flag = count % 100

        criterion_one, criterion_class = self.criterion_one, self.criterion_class
        p_code, b_code = self.p_code, self.b_code

        fake_imgs = self.fake_img[0]
        fake_mk = self.mk_img[0]

        # final image real/fake loss
        fake_logits = self.netsD[1](fake_imgs)[1]
        real_labels = torch.ones_like(fake_logits)
        errG = criterion_one(fake_logits, real_labels)
        errG_total += errG

        # Real/Fake loss for the real_img+b pair and real_img+p pair
        fake_bg = (torch.ones_like(fake_mk) - fake_mk) * fake_imgs
        fake_fg = fake_mk * fake_imgs

        fake_logits_b, fake_logits_p = self.bi_netD(fake_imgs, b_code, p_code, fake_mk)

        real_labels = torch.ones_like(fake_logits_b)

        errBiD_fake_b = criterion_one(fake_logits_b, real_labels)
        errBiD_fake_p = criterion_one(fake_logits_p, real_labels)
        errG_total += errBiD_fake_b + errBiD_fake_p

        # info loss for bg and fg
        with torch.no_grad():
            fake_pred_b = self.netsD[0](fake_bg)[0]
            fake_pred_p = self.netsD[1](fake_fg)[0]

        b_info_wt = 1.
        p_info_wt = 1.
        errG_info_b = criterion_class(fake_pred_b, torch.nonzero(b_code.long())[:,1]) * b_info_wt
        errG_info_p = criterion_class(fake_pred_p, torch.nonzero(p_code.long())[:,1]) * p_info_wt
        errG_total += errG_info_b + errG_info_p

        if flag == 0:
            summary_BiD_fake = summary.scalar('G_BiD_loss_fake_0', errBiD_fake_b.item())
            self.summary_writer.add_summary(summary_BiD_fake, count)
            summary_BiD_fake = summary.scalar('G_BiD_loss_fake_1', errBiD_fake_p.item())
            self.summary_writer.add_summary(summary_BiD_fake, count)
            summary_D_class = summary.scalar('Information_loss_0', errG_info_b.item())
            self.summary_writer.add_summary(summary_D_class, count)
            summary_D_class = summary.scalar('Information_loss_1', errG_info_p.item())
            self.summary_writer.add_summary(summary_D_class, count)
            summary_D = summary.scalar('G_loss', errG.item())
            self.summary_writer.add_summary(summary_D, count)

        # bg_mk = torch.ones_like(fake_mk) - fake_mk
        ms = fake_mk.size()
        min_fg_cvg = cfg.TRAIN.MIN_FG_CVG * ms[2] * ms[3]
        # min_bg_cvg = cfg.TRAIN.MIN_BG_CVG * ms[2] * ms[3]
        binary_loss = self.binarization_loss(fake_mk) * 1e1
        fg_cvg_loss = F.relu(min_fg_cvg - torch.sum(fake_mk, dim=(-1,-2))).mean() * 1e-2
        # bg_cvg_loss = F.relu(min_bg_cvg - torch.sum(bg_mk, dim=(-1,-2))).mean() * 0

        errG_total += binary_loss + fg_cvg_loss #+ bg_cvg_loss

        self.cl = fg_cvg_loss #+ bg_cvg_loss
        self.bl = binary_loss
        # self.ol = oob_loss

        errG_total.backward()
        for optG in self.optimizerG:
            optG.step()
        return errG_total


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


    def sample_pb_code(self, batch_size):
        has_underused_b = False
        if len(self.underused_b) != 0:
            has_underused_b = True

        rand_pids = list(np.random.randint(cfg.FG_CATEGORIES, size=batch_size))
        b_code = torch.torch.zeros([batch_size, cfg.BG_CATEGORIES]).cuda()
        p_code = torch.torch.zeros([batch_size, cfg.FG_CATEGORIES]).cuda()
        for i in range(batch_size):
            pid = rand_pids[i]

            if has_underused_b and torch.rand(1) < 0.6:
                bid = random.sample(self.underused_b, 1)[0]
            else:
                bid = random.sample(self.real_pb_pair[pid], 1)[0]
                # if sampled bid is overly used, try resample from not overly used
                if (self.mapped_b_count[bid] >= self.overused_thld) and \
                   (torch.rand(1) < 0.6):
                    bid = random.sample(self.not_overused_b, 1)[0]

            b_code[i, bid] = 1
            p_code[i, pid] = 1
        return p_code, b_code


    def train(self):
        self.netG, self.netsD, self.bi_netD, self.mkpred_net, start_count = load_network(self.gpus)
        newly_loaded = True
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD, self.optimizersBiD, self.optimizersMk = \
            define_optimizers(self.netG, self.netsD, self.bi_netD, self.mkpred_net)

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

        # maintain mapping balance
        self.mapped_b_count = {}
        for b in range(cfg.BG_CATEGORIES):
            self.mapped_b_count[b] = 0

        self.underused_b = set(range(cfg.BG_CATEGORIES))
        self.underuse_thld = 10

        self.not_overused_b = set(range(cfg.BG_CATEGORIES))
        self.overused_thld = 100

        store_len = 100
        self.real_pb_pair = {}
        for p in range(cfg.FG_CATEGORIES):
            rand_b = np.random.randint(cfg.BG_CATEGORIES, size=store_len).tolist()
            self.real_pb_pair[p] = rand_b
            for b in rand_b:
                self.mapped_b_count[b] += 1
                if (b in self.underused_b) and \
                   (self.mapped_b_count[b] >= self.underuse_thld):
                    self.underused_b.remove(b)

                if (b in self.not_overused_b) and \
                   (self.mapped_b_count[b] >= self.overused_thld):
                    self.not_overused_b.remove(b)

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

                    count += 1
                    self.real_cimgs = self.prepare_data(data)
                    bsz = self.real_cimgs.size(0)
                    self.p_code, self.b_code = self.sample_pb_code(bsz)
                    # Feedforward through Generator. Obtain stagewise fake images
                    noise.data.normal_(0, 1)
                    self.fake_img, self.raw_imgs, self.mk_img = self.netG(noise, self.p_code, self.b_code)

                    # Update Discriminator networks
                    self.train_mkpred_net(count)
                    errD_total = self.train_Dnet(count)
                    self.train_BiDnet(count)

                    # Update the Generator networks
                    errG_total = self.train_Gnet(count)

                    for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                        avg_p.mul_(0.999).add_(0.001, p.data)

                    newly_loaded = False
                    if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                        print("binary_loss: {}, cvg_loss: {}".format(self.bl.item(), self.cl.item()))
                        print(self.mapped_b_count)

                        backup_para = copy_G_params(self.netG)
                        if count % cfg.TRAIN.SAVEMODEL_INTERVAL == 0:
                            save_model(self.netG, avg_param_G, self.netsD, self.bi_netD, self.mkpred_net,
                                       self.real_pb_pair, count, self.model_dir, cur_depth)
                        # Save images
                        load_params(self.netG, avg_param_G)

                        with torch.no_grad():
                            fake_img, raw_imgs, mk_img = self.netG(fixed_noise, self.p_code, self.b_code)
                            real_pred_mk = self.mkpred_net(self.real_cimgs)

                        fake_fg = mk_img[0] * fake_img[0]
                        fake_bg = (torch.ones_like(mk_img[0]) - mk_img[0]) * fake_img[0]
                        real_fg = real_pred_mk * self.real_cimgs
                        real_bg = (torch.ones_like(real_pred_mk) - real_pred_mk) * self.real_cimgs
                        save_img_results((fake_img + raw_imgs + mk_img + [fake_fg, fake_bg] + [
                                          self.real_cimgs, real_pred_mk, real_fg, real_bg]),
                                          count, self.image_dir, self.summary_writer, cur_depth)
                        #
                        load_params(self.netG, backup_para)

                end_t = time.time()
                print('''[%d/%d][%d]Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                      % (epoch, max_epoch, num_batches,
                        errD_total.item(), errG_total.item(),
                        end_t - start_t))

            if not newly_loaded:
                save_model(self.netG, avg_param_G, self.netsD, self.bi_netD, self.mkpred_net,
                            count, self.model_dir, cur_depth)
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


            b = random.randint(0, cfg.BG_CATEGORIES-1)
            p = random.randint(0, cfg.SUPER_CATEGORIES-1)
            c = random.randint(0, cfg.FINE_GRAINED_CATEGORIES-1)
            bg_code = torch.zeros([self.batch_size, cfg.BG_CATEGORIES])
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
            bbgmk_li = []
            b_li = np.random.permutation(cfg.BG_CATEGORIES-1)
            p_li = np.random.permutation(cfg.SUPER_CATEGORIES-1)
            c_li = np.random.permutation(cfg.FINE_GRAINED_CATEGORIES-1)

            # b_li = np.array(range(0, cfg.BG_CATEGORIES))
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
            # c_li = np.array(range(98, 196))
            nrow = 20
            for k in range(1):
                b = b_li[k]
                p = p_li[k]
                c = c_li[k]

                for i in range(nrow):
                    bg_code = torch.zeros([self.batch_size, cfg.BG_CATEGORIES])
                    p_code = torch.zeros([self.batch_size, cfg.SUPER_CATEGORIES])
                    c_code = torch.zeros([self.batch_size, cfg.FINE_GRAINED_CATEGORIES])

                    # noise.data.normal_(0, 1)
                    # b = random.randint(0, cfg.BG_CATEGORIES-1)
                    p = i
                    # p = random.randint(0, cfg.SUPER_CATEGORIES-1)
                    # c = random.randint(0, cfg.FINE_GRAINED_CATEGORIES-1)
                    # b = b_li[i]
                    # p = p_li[i]
                    # c = c_li[i]
                    # b = c % cfg.BG_CATEGORIES
                    # p = int(c // 9.8)
                    # print('b:', b, 'p:', p, 'c:', c)
                    # p = i
                    for j in range(self.batch_size):
                        bg_code[j][b] = 1
                        p_code[j][p] = 1
                        c_code[j][c] = 1

                    with torch.no_grad():
                        fake_imgs, fg_imgs, mk_imgs, fgmk_imgs, _ = netG(
                            noise, c_code, p_code, bg_code)  # Forward pass through the generator

                    fg_mask = mk_imgs[0][0]
                    bg_mask = torch.ones_like(fg_mask) - fg_mask
                    bg_of_bg = bg_mask * fake_imgs[0][0]

                    bg_li.append(fake_imgs[0][0])
                    pf_li.append(fake_imgs[1][0])
                    cf_li.append(fake_imgs[2][0])
                    pk_li.append(mk_imgs[0][0])
                    ck_li.append(mk_imgs[1][0])
                    pfg_li.append(fg_imgs[0][0])
                    cfg_li.append(fg_imgs[1][0])
                    pfgmk_li.append(fgmk_imgs[0][0])
                    cfgmk_li.append(fgmk_imgs[1][0])
                    bbgmk_li.append(bg_of_bg)

            save_image(bg_li, self.save_dir, '0background', nrow, res)
            save_image(pf_li, self.save_dir, '1parent_final', nrow, res)
            save_image(cf_li, self.save_dir, '2child_final', nrow, res)
            save_image(pfg_li, self.save_dir, '3parent_foreground', nrow, res)
            save_image(cfg_li, self.save_dir, '4child_foreground', nrow, res)
            save_image(pk_li, self.save_dir, '5parent_mask', nrow, res)
            save_image(ck_li, self.save_dir, '6child_mask', nrow, res)
            save_image(pfgmk_li, self.save_dir,
                       '7parent_foreground_masked', nrow, res)
            save_image(cfgmk_li, self.save_dir,
                       '8child_foreground_masked', nrow, res)
            save_image(bbgmk_li, self.save_dir,
                       '9background_background_masked', nrow, res)

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
