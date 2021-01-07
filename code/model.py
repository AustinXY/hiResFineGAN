import sys
import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Upsample
import math

start_depth = cfg.TRAIN.START_DEPTH


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


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


def convlxl(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=13, stride=1,
                     padding=1, bias=False)


def child_to_parent(c_code):
    ratio = cfg.FINE_GRAINED_CATEGORIES / cfg.SUPER_CATEGORIES
    cid = torch.argmax(c_code, dim=1)
    pid = (cid / ratio).long()
    # print(torch.argmax(c_code,  dim=1))
    # print(pid)
    p_code = torch.zeros([c_code.size(0), cfg.SUPER_CATEGORIES]).cuda()
    for i in range(c_code.size(0)):
        p_code[i][pid[i]] = 1
    return p_code


def child_to_background(c_code):
    cid = torch.argmax(c_code,  dim=1)
    bid = cid % cfg.BG_CATEGORIES
    # print(bid)
    # sys.exit(0)
    b_code = torch.zeros([c_code.size(0), cfg.BG_CATEGORIES]).cuda()
    for i in range(c_code.size(0)):
        b_code[i][bid[i]] = 1
    return b_code

# def child_to_background_rand_b(c_code):
#     # cid = torch.argmax(c_code,  dim=1)
#
#     b_code = torch.zeros([c_code.size(0), bg_categories]).cuda()
#     for i in range(c_code.size(0)):
#         bid = torch.randint(0, bg_categories-1, ())
#         b_code[i][bid] = 1
#     return b_code

# def parent_to_background(p_code):
#     pid = torch.argmax(p_code,  dim=1)
#     bg_categories = cfg.SUPER_CATEGORIES // cfg.NUM_P_PER_B * cfg.NUM_B_PER_P
#     b_code = torch.zeros([p_code.size(0), bg_categories]).cuda()
#     for i in range(p_code.size(0)):
#         b_bin = pid[i] // cfg.NUM_P_PER_B
#         bid = torch.randint(b_bin * cfg.NUM_B_PER_P, (b_bin+1) * cfg.NUM_B_PER_P, (1,))
#         b_code[i][bid] = 1
#     return b_code

# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


def sameBlock(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

# Keep the spatial size


def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, c_flag):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.c_flag = c_flag

        if self.c_flag == 1:  # parent
            self.in_dim = cfg.GAN.Z_DIM + cfg.SUPER_CATEGORIES
        elif self.c_flag == 2:  # child
            self.in_dim = cfg.GAN.Z_DIM + cfg.FINE_GRAINED_CATEGORIES
        else:  # bg
            self.in_dim = cfg.GAN.Z_DIM + cfg.BG_CATEGORIES

        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # self.upsample5 = upBlock(ngf // 16, ngf // 16)

    def forward(self, z_code, code):
        in_code = torch.cat((code, z_code), 1)
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)  # 1024 * 4 * 4
        out_code = self.upsample1(out_code)              # 512 * 8 * 8
        out_code = self.upsample2(out_code)              # 256 * 16 * 16
        out_code = self.upsample3(out_code)              # 128 * 32 * 32
        # out_code = self.upsample4(out_code)
        # out_code = self.upsample5(out_code)

        return out_code


class NEXT_STAGE_G_SAME(nn.Module):
    def __init__(self, ngf, use_hrc=1, num_residual=cfg.GAN.R_NUM):
        super().__init__()
        self.gf_dim = ngf
        if use_hrc == 1:  # For parent stage
            self.ef_dim = cfg.SUPER_CATEGORIES
        elif use_hrc == 2:  # For child
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES
        else:
            self.ef_dim = cfg.BG_CATEGORIES

        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim
        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.samesample = sameBlock(ngf, ngf // 2)

    def forward(self, h_code, code):
        s_size = h_code.size(2)
        code = code.view(-1, self.ef_dim, 1, 1)
        code = code.repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((code, h_code), 1)
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code


class NEXT_STAGE_G_UP(nn.Module):
    def __init__(self, ngf, use_hrc=1, num_residual=cfg.GAN.R_NUM):
        super().__init__()
        self.gf_dim = ngf
        if use_hrc == 1:  # For parent stage
            self.ef_dim = cfg.SUPER_CATEGORIES
        elif use_hrc == 2:  # For child
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES
        else:  # bg
            self.ef_dim = cfg.BG_CATEGORIES

        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim
        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        # self.samesample = sameBlock(ngf, ngf // 2)
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, code):
        s_size = h_code.size(2)
        code = code.view(-1, self.ef_dim, 1, 1)
        code = code.repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((code, h_code), 1)
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.upsample(out_code)
        return out_code


class TO_RGB_LAYER(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class TO_GRAY_LAYER(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 1),
            nn.Sigmoid()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.gf_dim = 64
        self.define_module()

    def define_module(self):
        ngf = self.gf_dim

        self.bg_code_init = INIT_STAGE_G(ngf * 8, c_flag=0)
        self.bg_code_net = nn.ModuleList([NEXT_STAGE_G_SAME(ngf, use_hrc=0)])
        self.bg_img_net = nn.ModuleList([TO_RGB_LAYER(ngf // 2)])

        self.p_code_init = INIT_STAGE_G(ngf * 8, c_flag=1)
        self.p_code_net = nn.ModuleList([NEXT_STAGE_G_SAME(ngf, use_hrc=1)])

        self.p_fg_net = nn.ModuleList([TO_RGB_LAYER(ngf // 2)])
        self.p_mk_net = nn.ModuleList([TO_GRAY_LAYER(ngf // 2)])

        self.c_code_net = nn.ModuleList([NEXT_STAGE_G_SAME(ngf // 2, use_hrc=2)])
        self.c_fg_net = nn.ModuleList([TO_RGB_LAYER(ngf // 4)])
        self.c_mk_net = nn.ModuleList([TO_GRAY_LAYER(ngf // 4)])

        ngf = ngf // 2

        for _ in range(start_depth):
            self.bg_code_net.append(NEXT_STAGE_G_UP(ngf, use_hrc=0))
            self.bg_img_net.append(TO_RGB_LAYER(ngf // 2))

            self.p_code_net.append(NEXT_STAGE_G_UP(ngf, use_hrc=1))

            self.p_fg_net.append(TO_RGB_LAYER(ngf // 2))
            self.p_mk_net.append(TO_GRAY_LAYER(ngf // 2))

            self.c_code_net.append(NEXT_STAGE_G_SAME(ngf // 2, use_hrc=2))
            self.c_fg_net.append(TO_RGB_LAYER(ngf // 4))
            self.c_mk_net.append(TO_GRAY_LAYER(ngf // 4))

            ngf = ngf // 2

        self.gf_dim = ngf
        self.cur_depth = start_depth
        self.recon_net = RECON_NET()

    def inc_depth(self):
        # oneline apply should work??
        ngf = self.gf_dim

        self.bg_code_net.append(NEXT_STAGE_G_UP(
            ngf, use_hrc=0).apply(weights_init))
        self.bg_img_net.append(TO_RGB_LAYER(ngf // 2).apply(weights_init))

        self.p_code_net.append(NEXT_STAGE_G_UP(
            ngf, use_hrc=1).apply(weights_init))

        self.p_fg_net.append(TO_RGB_LAYER(ngf // 2).apply(weights_init))
        self.p_mk_net.append(TO_GRAY_LAYER(ngf // 2).apply(weights_init))

        self.c_code_net.append(NEXT_STAGE_G_SAME(
            ngf // 2, use_hrc=2).apply(weights_init))
        self.c_fg_net.append(TO_RGB_LAYER(ngf // 4).apply(weights_init))
        self.c_mk_net.append(TO_GRAY_LAYER(ngf // 4).apply(weights_init))

        ngf = ngf // 2
        self.gf_dim = ngf
        self.cur_depth += 1

    def forward(self, z_code, c_code, p_code=None, bg_code=None, alpha=None):

        fake_imgs = []  # Will contain [background image, parent image, child image]
        fg_imgs = []  # Will contain [parent foreground, child foreground]
        mk_imgs = []  # Will contain [parent mask, child mask]
        fg_mk = []  # Will contain [masked parent foreground, masked child foreground]

        if cfg.TIED_CODES:
            # Obtaining the parent code from child code
            p_code = child_to_parent(c_code)
            # bg_code = child_to_background(c_code)
            bg_code = c_code


        h_code_bg = self.bg_code_init(z_code, bg_code)
        h_code_p = self.p_code_init(z_code, p_code)

        for i in range(self.cur_depth + 1):
            _bg_code_net = self.bg_code_net[i]
            _bg_img_net = self.bg_img_net[i]
            _p_code_net = self.p_code_net[i]
            _p_fg_net = self.p_fg_net[i]
            _p_mk_net = self.p_mk_net[i]
            _c_code_net = self.c_code_net[i]
            _c_fg_net = self.c_fg_net[i]
            _c_mk_net = self.c_mk_net[i]

            h_code_bg = _bg_code_net(h_code_bg, bg_code)

            fake_img1 = _bg_img_net(h_code_bg)
            if i == self.cur_depth and i != start_depth and alpha < 1:
                prev_fake_img1 = fake_imgs[(i-1)*3]
                prev_fake_img1 = F.upsample(
                    prev_fake_img1, scale_factor=2)  # mode='nearest'
                fake_img1 = (1 - alpha) * prev_fake_img1 + alpha * fake_img1

            h_code_p = _p_code_net(h_code_p, p_code)

            fake_img2_fg = _p_fg_net(h_code_p)  # Parent foreground
            fake_img2_mk = _p_mk_net(h_code_p)  # Parent mask
            if i == self.cur_depth and i != start_depth and alpha < 1:
                prev_fake_img2_fg = fg_imgs[(i-1)*2]
                prev_fake_img2_fg = F.upsample(
                    prev_fake_img2_fg, scale_factor=2)  # mode='nearest'
                fake_img2_fg = (1 - alpha) * \
                    prev_fake_img2_fg + alpha * fake_img2_fg
                prev_fake_img2_mk = mk_imgs[(i-1)*2]
                prev_fake_img2_mk = F.upsample(
                    prev_fake_img2_mk, scale_factor=2)  # mode='nearest'
                fake_img2_mk = (1 - alpha) * \
                    prev_fake_img2_mk + alpha * fake_img2_mk

            h_code_c = _c_code_net(h_code_p, c_code)

            fake_img3_fg = _c_fg_net(h_code_c)  # Child foreground
            fake_img3_mk = _c_mk_net(h_code_c)  # Child mask
            # fake_img3_mk = fake_img2_mk * fake_img3_mk
            # fake_img3_mk = (1 - 1) * fake_img3_mk + 1 * (fake_img2_mk * fake_img3_mk)
            if i == self.cur_depth and i != start_depth and alpha < 1:
                prev_fake_img3_fg = fg_imgs[(i-1)*2+1]
                prev_fake_img3_fg = F.upsample(
                    prev_fake_img3_fg, scale_factor=2)  # mode='nearest'
                fake_img3_fg = (1 - alpha) * \
                    prev_fake_img3_fg + alpha * fake_img3_fg
                prev_fake_img3_mk = mk_imgs[(i-1)*2+1]
                prev_fake_img3_mk = F.upsample(
                    prev_fake_img3_mk, scale_factor=2)  # mode='nearest'
                fake_img3_mk = (1 - alpha) * \
                    prev_fake_img3_mk + alpha * fake_img3_mk

            if i == self.cur_depth:
                fake_imgs.append(fake_img1)
                ones_mask_p = torch.ones_like(fake_img2_mk)
                opp_mask_p = ones_mask_p - fake_img2_mk
                fg_masked2 = torch.mul(fake_img2_fg, fake_img2_mk)
                fg_mk.append(fg_masked2)
                bg_masked2 = torch.mul(fake_img1, opp_mask_p)
                fake_img2_final = fg_masked2 + bg_masked2  # Parent image
                fake_imgs.append(fake_img2_final)
                fg_imgs.append(fake_img2_fg)
                mk_imgs.append(fake_img2_mk)
                ones_mask_c = torch.ones_like(fake_img3_mk)
                opp_mask_c = ones_mask_c - fake_img3_mk
                fg_masked3 = torch.mul(fake_img3_fg, fake_img3_mk)
                fg_mk.append(fg_masked3)
                bg_masked3 = torch.mul(fake_img2_final, opp_mask_c)
                fake_img3_final = fg_masked3 + bg_masked3  # Child image
                fake_imgs.append(fake_img3_final)
                fg_imgs.append(fake_img3_fg)
                mk_imgs.append(fake_img3_mk)

        return fake_imgs, fg_imgs, mk_imgs, fg_mk, [p_code, bg_code]

def recon_mask_info(fg_attn, fg_mk):
    ms = fg_mk.size()
    return torch.bmm(fg_mk.view(ms[0], 1, ms[2]*ms[3]), fg_attn).view(ms)

# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, padding, dilation, groups, bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.AvgPool2d(2)
    )
    return block


class MnetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(
            1, 1, kernel_size, stride, padding, dilation, groups, False)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        """
        input is regular tensor with shape N*C*H*W
        mask has to have 1 channel N*1*H*W
        """
        output = self.input_conv(input)
        if mask != None:
            mask = self.mask_conv(mask)
        return output, mask


class downBlock_mnet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = MnetConv(in_channels, out_channels,
                             kernel_size, stride, padding, dilation, groups, bias)
        # self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input, mask=None):
        """
        input is regular tensor with shape N*C*H*W
        mask has to have 1 channel N*1*H*W
        """
        output, mask = self.conv(input, mask)
        # output = self.bn(output)
        output = F.leaky_relu(output, 0.2, inplace=True)
        output = F.avg_pool2d(output, 2)
        if mask != None:
            mask = F.avg_pool2d(mask, 2)
        return output, mask


def fromRGB_layer(out_planes):
    layer = nn.Sequential(
        nn.Conv2d(3, out_planes, 1, 1, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return layer


def fromGRAY_layer(out_planes):
    layer = nn.Sequential(
        nn.Conv2d(1, out_planes, 1, 1, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return layer


class RECON_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.df_dim = 64
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        self.from_gray = fromGRAY_layer(ndf)
        self.downblock1 = downBlock(ndf, ndf * 2, 3, 1, 1)
        self.sameblock1 = sameBlock(ndf * 2, ndf * 2)
        self.downblock2 = downBlock(ndf * 2, ndf * 4, 3, 1, 1)
        self.sameblock2 = sameBlock(ndf * 4, ndf * 4)

        self.upblock1 = upBlock(ndf * 4, ndf * 2)
        self.sameblock3 = sameBlock(ndf * 2, ndf * 2)
        self.upblock2 = upBlock(ndf * 2, ndf)
        self.sameblock4 = sameBlock(ndf, ndf)
        self.to_gray = TO_GRAY_LAYER(ndf)

    def forward(self, mask_info):
        recon_mask = self.from_gray(mask_info)
        recon_mask = self.downblock1(recon_mask)
        recon_mask = self.sameblock1(recon_mask)
        recon_mask = self.downblock2(recon_mask)
        recon_mask = self.sameblock2(recon_mask)
        recon_mask = self.upblock1(recon_mask)
        recon_mask = self.sameblock3(recon_mask)
        recon_mask = self.upblock2(recon_mask)
        recon_mask = self.sameblock4(recon_mask)
        recon_mask = self.to_gray(recon_mask)
        return recon_mask


class D_NET_PC_BASE(nn.Module):
    def __init__(self, stg_no, ndf):
        super().__init__()
        self.df_dim = ndf
        self.stg_no = stg_no
        if self.stg_no == 1:
            self.ef_dim = cfg.SUPER_CATEGORIES
        elif self.stg_no == 2:
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES
        else:
            self.ef_dim = cfg.BG_CATEGORIES

        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim

        self.downblock1 = downBlock(ndf, ndf * 2, 3, 1, 1)
        self.downblock2 = downBlock(ndf * 2, ndf * 2, 3, 1, 1)
        self.downblock3 = downBlock(ndf * 2, ndf * 2, 3, 1, 1)
        # self.conv = Block3x3_leakRelu(ndf, ndf * 2)
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 2, efg, kernel_size=4, stride=4))
        self.jointConv = Block3x3_leakRelu(ndf * 2, ndf * 2)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_code):
        x_code = self.downblock1(x_code)  # 512 * 16 * 16
        x_code = self.downblock2(x_code)  # 512 * 8 * 8
        x_code = self.downblock3(x_code)  # 512 * 4 * 4
        # x_code = self.conv(x_code)
        h_c_code = self.jointConv(x_code)
        # Predicts the parent code and child code in parent and child stage respectively
        code_pred = self.logits(h_c_code)
        # This score is not used in parent stage while training
        rf_score = self.uncond_logits(x_code)
        return [code_pred.view(-1, self.ef_dim), rf_score.view(-1)]


class D_NET_PC(nn.Module):
    def __init__(self, stg_no):
        super().__init__()
        self.df_dim = 256
        self.stg_no = stg_no

        self.define_module()

    def define_module(self):
        ndf = self.df_dim

        self.from_RGB_net = nn.ModuleList([fromRGB_layer(ndf)])
        self.down_net = nn.ModuleList([D_NET_PC_BASE(self.stg_no, ndf)])
        ndf = ndf // 2

        for _ in range(start_depth):
            self.from_RGB_net.append(fromRGB_layer(ndf))
            self.down_net.append(downBlock(ndf, ndf * 2, 3, 1, 1))
            ndf = ndf // 2

        self.df_dim = ndf
        self.cur_depth = start_depth

    def inc_depth(self):
        ndf = self.df_dim
        self.from_RGB_net.append(fromRGB_layer(ndf).apply(weights_init))
        self.down_net.append(
            downBlock(ndf, ndf * 2, 3, 1, 1).apply(weights_init))
        ndf = ndf // 2
        self.df_dim = ndf
        self.cur_depth = self.cur_depth + 1

    def forward(self, x_var, alpha=None, mask=None):
        x_code = self.from_RGB_net[self.cur_depth](x_var)
        for i in range(self.cur_depth, -1, -1):
            x_code = self.down_net[i](x_code)
            if i == self.cur_depth and i != start_depth and alpha < 1:
                y_var = F.avg_pool2d(x_var, 2)
                y_code = self.from_RGB_net[i-1](y_var)
                x_code = (1 - alpha) * y_code + alpha * x_code

        code_pred = x_code[0]
        rf_score = x_code[1]
        return [code_pred, rf_score]
