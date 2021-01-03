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


from model import G_NET, D_NET, PBMatcher


start_depth = cfg.TRAIN.START_DEPTH
end_depth = cfg.TRAIN.END_DEPTH
batchsize_per_depth = cfg.TRAIN.BATCHSIZE_PER_DEPTH
blend_epochs_per_depth = cfg.TRAIN.BLEND_EPOCHS_PER_DEPTH
stable_epochs_per_depth = cfg.TRAIN.STABLE_EPOCHS_PER_DEPTH

# ################## Shared functions ###################

def child_to_parent(child_c_code):
    ratio = cfg.FINE_GRAINED_CATEGORIES / cfg.SUPER_CATEGORIES
    arg_parent = torch.argmax(child_c_code,  dim = 1) / ratio
    parent_c_code = torch.zeros([child_c_code.size(0), cfg.SUPER_CATEGORIES]).cuda()
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

def load_network(gpus, load_D=True):
    matcher = PBMatcher()
    matcher.apply(weights_init)
    matcher = torch.nn.DataParallel(matcher, device_ids=gpus)

    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    # print(netG)

    netsD = []
    if load_D:
        netsD.append(D_NET(0))
        netsD.append(D_NET(1))
        netsD.append(D_NET(2))

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

    if load_D:
        if cfg.TRAIN.NET_D != '':
            for i in range(len(netsD)):
                print('Load %s%d_%s.pth' % (cfg.TRAIN.NET_D, i, _depth))
                state_dict = torch.load('%s%d_%s.pth' % (cfg.TRAIN.NET_D, i, _depth))
                netsD[i].load_state_dict(state_dict)

    if cfg.TRAIN.MATCHER != '':
        print('Load ', cfg.TRAIN.MATCHER)
        state_dict = torch.load(cfg.TRAIN.MATCHER)
        matcher.load_state_dict(state_dict)

    if cfg.CUDA:
        matcher.cuda()
        netG.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()

    return netG, netsD, matcher, count


def define_optimizers(netG, netsD, matcher):
    opt_matcher = optim.Adam(matcher.parameters(),
                            lr=1e-4,
                            betas=(0.5, 0.999))

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

    return optimizerG, optimizersD, opt_matcher


def save_model(netG, avg_param_G, netsD, matcher, epoch, model_dir, cur_depth):
    load_params(netG, avg_param_G)
    torch.save(
        netG.state_dict(),
        '%s/netG_%d_depth%d.pth' % (model_dir, epoch, cur_depth))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(netD.state_dict(),
            '%s/netD%d_depth%d.pth' % (model_dir, i, cur_depth))

    torch.save(matcher.state_dict(),
            '%s/matcher_depth%d.pth' % (model_dir, cur_depth))
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


    def train_Dnet(self, count):
        flag = count % 100
        criterion_one = self.criterion_one

        netD, optD = self.netsD[2], self.optimizersD[0]

        real_imgs = self.real_cimgs

        fake_imgs = self.fake_imgs[2]
        netD.zero_grad()
        real_logits = netD(real_imgs, self.alpha)

        fake_labels = torch.zeros_like(real_logits[1])
        real_labels = torch.ones_like(real_logits[1])

        fake_logits = netD(fake_imgs.detach(), self.alpha)

        errD_real = criterion_one(real_logits[1], real_labels) # Real/Fake loss for the real image
        errD_fake = criterion_one(fake_logits[1], fake_labels) # Real/Fake loss for the fake image
        errD = errD_real + errD_fake

        errD.backward()
        optD.step()

        if flag == 0:
            summary_D = summary.scalar('D_loss%d' % 2, errD.item())
            self.summary_writer.add_summary(summary_D, count)
            summary_D_real = summary.scalar('D_loss_real_%d' % 2, errD_real.item())
            self.summary_writer.add_summary(summary_D_real, count)
            summary_D_fake = summary.scalar('D_loss_fake_%d' % 2, errD_fake.item())
            self.summary_writer.add_summary(summary_D_fake, count)

        return errD


    def train_Gnet(self, count):
        self.netG.zero_grad()
        for i in range(len(self.netsD)):
             self.netsD[i].zero_grad()

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
        for i in range(len(self.netsD)):
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
                pred_b = self.netsD[i](bg_of_bg, self.alpha)
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
        binary_loss = self.binarization_loss(fg_mk) * 1e1
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


    def train_matcher(self, count):
        # default_bump = 1e-7
        default_bump_thrld = 0.3

        self.opt_matcher.zero_grad()

        b_prob = self.b_prob
        pid = torch.argmax(self.p_code,  dim=-1)
        bid = torch.argmax(self.b_code,  dim=-1)
        bid_front = torch.argmax(self.b_code_front,  dim=-1)
        bid_back = torch.argmax(self.b_code_back,  dim=-1)

        evaluator = self.netsD[2]

        fake_imgs = self.fake_imgs[2]
        fake_imgs_front = self.fake_imgs_front[2]
        fake_imgs_back = self.fake_imgs_back[2]

        errG_total = 0

        with torch.no_grad():
            eval = evaluator(fake_imgs, self.alpha)[1]
            eval_front = evaluator(fake_imgs_front, self.alpha)[1]
            eval_back = evaluator(fake_imgs_back, self.alpha)[1]

        # front_cmp = (F.relu(eval - eval_front))**2
        # back_cmp = (F.relu(eval_back - eval))**2

        # print(eval)
        # print(eval_front)
        # print(eval_back)

        # print(front_cmp)
        # print(back_cmp)

        temp_mat = torch.zeros_like(self.b_prob)
        for i in range(temp_mat.size(0)):
            v = self.pb_eval_update(pid[i], bid[i], eval[i])
            vf = self.pb_eval_update(pid[i], bid_front[i], eval_front[i])
            vb = self.pb_eval_update(pid[i], bid_back[i], eval_back[i])

            default_bump = (F.relu(default_bump_thrld - self.selected_b_prob[i]))**2 * 1e-4
            temp_mat[i][bid[i]] = -default_bump - F.relu(v - vf) + F.relu(vb - v)
            # temp_mat[i][bid_front[i]] = front_cmp[i]
            # temp_mat[i][bid_back[i]] = -back_cmp[i]

        matcher_loss = torch.sum(temp_mat * b_prob) * 1e-2

        # matcher_loss.backward()

        # print(self.matcher.module.predictor[-2].weight.grad)

        errG_total += matcher_loss

        if count % 100 == 0:
            summary_D = summary.scalar('Matcher_loss', matcher_loss.item())
            self.summary_writer.add_summary(summary_D, count)

        # matching sparcity loss: don't want too many p to match to same b
        batch_prob = torch.zeros(b_prob.size(1)).cuda()
        p_li = []
        for i in range(b_prob.size(0)):
            if pid[i] not in p_li:
                p_li.append(pid[i])
                batch_prob += b_prob[i]

            # if pid[i] == 0:
            #     self.p0_prob = b_prob[i]
            #     for j in range(cfg.BG_CATEGORIES):
            #         summary_D = summary.scalar('p0_b%d_prob' % j, b_prob[i][j].item())
            #         self.summary_writer.add_summary(summary_D, count)

        target = torch.ones_like(batch_prob) * (0.6)
        sparsity_loss = torch.sum(F.relu(batch_prob - target)) * 1e-6

        errG_total += sparsity_loss

        self.sl = sparsity_loss
        self.ml = matcher_loss

        errG_total.backward()
        self.opt_matcher.step()
        return matcher_loss

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

    def parent_to_bg(self, p_code, cmp_rank=None):
        '''
        input:
        cmp_rank: compatibility rank, should be within [0, cfg.BG_CATEGORIES-1]
                  if it is not none, return b_code of the given rank

        output:
        b_code: sampled b_code
        b_code_front: b_code with larger b_prob
        b_code_back: b_code with smaller b_prob
        '''

        b_prob = self.matcher(p_code)
        b_code = torch.zeros([p_code.size(0), cfg.BG_CATEGORIES]).cuda()
        sorted_b_prob, sorted_bid = torch.sort(b_prob, dim=-1, descending=True)

        self.max_b_prob = sorted_b_prob[:, 0].detach()

        if cmp_rank is not None:
            for i in range(p_code.size(0)):
                max_bid = sorted_bid[i][cmp_rank]
                b_code[i][max_bid] = 1

            return b_code, sorted_b_prob[:, cmp_rank], None, None, None


        b_code_front = torch.zeros_like(b_code)
        b_code_back = torch.zeros_like(b_code)
        selected_b_prob = torch.zeros(p_code.size(0)).cuda()

        b_prob_acc = sorted_b_prob
        # accumulate probability
        for i in range(1, b_prob_acc.size(1)):
            b_prob_acc[:, i] = b_prob_acc[:, i-1] + b_prob_acc[:, i]

        rnd = torch.rand((p_code.size(0), 1)).cuda()
        for i in range(p_code.size(0)):
            _bid = torch.nonzero(b_prob_acc[i] >= rnd[i], as_tuple=True)[0][0]
            _bid_front = _bid - 1
            if _bid_front < 0:
                _bid_front = 0
            _bid_back = _bid + 1
            if _bid_back > cfg.BG_CATEGORIES-1:
                _bid_back = cfg.BG_CATEGORIES-1

            bid = sorted_bid[i][_bid]
            bid_front = sorted_bid[i][_bid_front]
            bid_back = sorted_bid[i][_bid_back]

            b_code[i][bid] = 1
            b_code_front[i][bid_front] = 1
            b_code_back[i][bid_back] = 1

            selected_b_prob[i] = b_prob[i][bid]

        return b_code, selected_b_prob, b_code_front, b_code_back, b_prob

    def pb_eval_update(self, _pid, _bid, eval):
        pid = _pid.item()
        bid = _bid.item()
        old = self.pb_eval_record[pid][bid][0].pop(0)
        self.pb_eval_record[pid][bid][1] += eval - old
        self.pb_eval_record[pid][bid][0].append(eval)
        return self.pb_eval_record[pid][bid][1] / self.record_len

    # def pb_eval(self, pid, bid):
    #     return self.pb_eval_record[pid][bid][1] / self.record_len

    def train(self):
        self.netG, self.netsD, self.matcher, start_count = load_network(self.gpus)
        newly_loaded = True
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD, self.opt_matcher = \
            define_optimizers(self.netG, self.netsD, self.matcher)

        self.criterion = nn.BCELoss(reduce=False)
        self.criterion_one = nn.BCELoss()
        self.criterion_class = nn.CrossEntropyLoss()

        nz = cfg.GAN.Z_DIM

        self.p0_prob = None

        self.record_len = 20
        self.pb_eval_record = {}
        for pid in range(cfg.SUPER_CATEGORIES):
            self.pb_eval_record[pid] = {}
            for bid in range(cfg.BG_CATEGORIES):
                self.pb_eval_record[pid][bid] = [[0.] * self.record_len, 0.]


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
                    self.real_cimgs, self.c_code = self.prepare_data(data)

                    # Obtain the parent code given the child code
                    self.p_code = child_to_parent(self.c_code)
                    self.b_code, self.selected_b_prob, self.b_code_front, self.b_code_back, self.b_prob = self.parent_to_bg(self.p_code)

                    # Feedforward through Generator. Obtain stagewise fake images
                    noise.data.normal_(0, 1)
                    self.fake_imgs, self.fg_imgs, self.mk_imgs, self.fg_mk = self.netG(noise, self.c_code, self.p_code, self.b_code, self.alpha)

                    # Update Discriminator networks
                    errD_total = self.train_Dnet(count)

                    # Update the Generator networks
                    errG_total = self.train_Gnet(count)
                    for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                        avg_p.mul_(0.999).add_(0.001, p.data)

                    with torch.no_grad():
                        self.fake_imgs, _, _, _ = self.netG(noise, self.c_code, self.p_code, self.b_code, self.alpha)
                        self.fake_imgs_front, _, _, _ = self.netG(noise, self.c_code, self.p_code, self.b_code_front, self.alpha)
                        self.fake_imgs_back, _, _, _ = self.netG(noise, self.c_code, self.p_code, self.b_code_back, self.alpha)

                    self.train_matcher(count)

                    newly_loaded = False
                    if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                        print("max probs: {}\nsparsity loss: {}, matcher loss: {}".
                              format(self.max_b_prob, self.sl.item(), self.ml.item()))
                        if self.p0_prob is not None:
                            print(self.p0_prob)
                        # print("binary_loss: {}, cvg_loss: {}, oob_loss: {}".
                        #       format(self.bl.item(), self.cl.item(), self.ol.item()))

                        backup_para = copy_G_params(self.netG)
                        if count % cfg.TRAIN.SAVEMODEL_INTERVAL == 0:
                            save_model(self.netG, avg_param_G, self.netsD, self.matcher, count, self.model_dir, cur_depth)
                        # Save images
                        load_params(self.netG, avg_param_G)

                        b_code, _, _, _, _ = self.parent_to_bg(self.p_code, cmp_rank=0)

                        fake_imgs, fg_imgs, mk_imgs, fg_mk = self.netG(
                            fixed_noise, self.c_code, self.p_code, b_code, self.alpha)

                        bg_img = fake_imgs[0]
                        bg_mask = torch.ones_like(mk_imgs[0]) - mk_imgs[0]
                        bg_of_bg = bg_mask * bg_img
                        save_img_results((fake_imgs + fg_imgs + mk_imgs + fg_mk + [bg_of_bg]),
                                         count, self.image_dir, self.summary_writer, cur_depth)
                        #
                        load_params(self.netG, backup_para)

                end_t = time.time()
                print('''[%d/%d][%d]Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                      % (epoch, max_epoch, num_batches,
                        errD_total.item(), errG_total.item(),
                        end_t - start_t))

            if not newly_loaded:
                save_model(self.netG, avg_param_G, self.netsD, self.matcher, count, self.model_dir, cur_depth)
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

    def parent_to_bg(self, p_code, sample_by_prob=True):
        b_prob = self.matcher(p_code)
        b_code = torch.zeros([p_code.size(0), cfg.BG_CATEGORIES]).cuda()

        if not sample_by_prob:
            sorted_b_prob, idx = torch.sort(b_prob, dim=-1, descending=True)
            selected_prob = sorted_b_prob[:, 0]
            max_prob_bid = idx[:, 0]
            for i in range(p_code.size(0)):
                bid = max_prob_bid[i]
                b_code[i][bid] = 1

        else:
            b_prob_acc = b_prob.clone()
            selected_prob = torch.zeros(p_code.size(0)).cuda()

            # accumulate probability
            for i in range(1, b_prob_acc.size(1)):
                b_prob_acc[:, i] = b_prob_acc[:, i-1] + b_prob_acc[:, i]

            rnd = torch.rand((p_code.size(0), 1)).cuda()
            # print(rnd)
            for i in range(p_code.size(0)):
                bid = torch.nonzero(
                    b_prob_acc[i] >= rnd[i], as_tuple=True)[0][0]
                b_code[i][bid] = 1
                selected_prob[i] = b_prob[i][bid]

        return b_code, selected_prob, b_prob

    def evaluate_finegan(self):
        random.seed(datetime.now())
        torch.manual_seed(random.randint(0, 9999))
        # torch.manual_seed(2)

        depth = cfg.TRAIN.START_DEPTH
        res = 32 * 2 ** depth
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for model not found!')
        else:
            # Build and load the generator
            netG, netsD, self.matcher, evaluator, _ = load_network(
                self.gpus, load_D=False)
            # netG = G_NET()
            # print(netG)
            # netG.apply(weights_init)
            # netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            # model_dict = netG.state_dict()

            # state_dict = \
            #     torch.load(cfg.TRAIN.NET_G,
            #                map_location=lambda storage, loc: storage)

            # state_dict = {k: v for k, v in state_dict.items()
            #               if k in model_dict}

            # model_dict.update(state_dict)
            # netG.load_state_dict(model_dict)
            # print('Load ', cfg.TRAIN.NET_G)

            # Uncomment this to print Generator layers
            # print(netG)

            nz = cfg.GAN.Z_DIM
            noise = torch.FloatTensor(1, nz)

            noise.data.normal_(0, 1)
            # noise = noise.repeat(1, 1)

            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            netG.eval()
            self.matcher.eval()
            evaluator.eval()

            b = random.randint(0, cfg.BG_CATEGORIES-1)
            p = random.randint(0, cfg.SUPER_CATEGORIES-1)
            c = random.randint(0, cfg.FINE_GRAINED_CATEGORIES-1)
            b_code = torch.zeros([1, cfg.BG_CATEGORIES])
            p_code = torch.zeros([1, cfg.SUPER_CATEGORIES])
            c_code = torch.zeros([1, cfg.FINE_GRAINED_CATEGORIES])

            bg_li = []
            pf_li = []
            cf_li = []
            pk_li = []
            ck_li = []
            pfg_li = []
            cfg_li = []
            pfgmk_li = []
            cfgmk_li = []
            # b_li = np.random.permutation(cfg.BG_CATEGORIES-1)
            # p_li = np.random.permutation(cfg.SUPER_CATEGORIES-1)
            # c_li = np.random.permutation(cfg.FINE_GRAINED_CATEGORIES-1)

            # c_li = p_c_dict[19]
            # c_li = np.array(range(0, 98))
            c_li = [0, 50]
            p_li = list(range(0, 20))
            b_li = [0]

            tie_code = False

            if tie_code:
                l = len(c_li)
            else:
                l = max(len(c_li), len(p_li), len(b_li))

            nrow = 10
            for k in range(1):
                b = b_li[k]
                p = p_li[k]
                c = c_li[k]

                for i in range(l):

                    # noise.data.normal_(0, 1)

                    if tie_code:
                        c_code = torch.zeros([1, cfg.FINE_GRAINED_CATEGORIES])
                        c = c_li[i]
                        c_code[0][c] = 1

                        p_code = child_to_parent(c_code)
                        p = torch.argmax(p_code,  dim=-1).item()

                        b_code, selected_prob, b_prob = self.parent_to_bg(p_code)
                        max_b_code, max_b_prob, _ = self.parent_to_bg(p_code, sample_by_prob=False)

                        b = torch.argmax(b_code,  dim=-1).item()
                        b_code = max_b_code
                    else:
                        # b = b_li[i]
                        p = p_li[i]
                        # c = c_li[i]

                        b_code = torch.zeros([1, cfg.BG_CATEGORIES])
                        p_code = torch.zeros([1, cfg.SUPER_CATEGORIES])
                        c_code = torch.zeros([1, cfg.FINE_GRAINED_CATEGORIES])

                        c_code[0][c] = 1
                        p_code[0][p] = 1
                        b_code[0][b] = 1

                    # sorted_b_prob, idx = torch.sort(b_prob, dim=-1, descending=True)

                    # print('c: {}, p: {}, b: {}'.format(c, p, b))

                    # print(sorted_b_prob[0][0:5])
                    # print(idx[0][0:5])

                    # sys.exit(0)

                    fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(
                        noise, c_code, p_code, b_code)  # Forward pass through the generator
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
