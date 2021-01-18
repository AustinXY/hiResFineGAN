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

        if self.c_flag == 1:  # fg
            self.in_dim = cfg.GAN.Z_DIM + cfg.FG_CATEGORIES
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
        if use_hrc == 1:  # For fg stage
            self.ef_dim = cfg.FG_CATEGORIES
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
        if use_hrc == 1:  # For fg stage
            self.ef_dim = cfg.FG_CATEGORIES
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

        self.b_code_init = INIT_STAGE_G(ngf * 8, c_flag=0)
        self.b_code_net = nn.ModuleList([NEXT_STAGE_G_SAME(ngf, use_hrc=0)])
        self.b_img_net = nn.ModuleList([TO_RGB_LAYER(ngf // 2)])

        self.p_code_init = INIT_STAGE_G(ngf * 8, c_flag=1)
        self.p_code_net = nn.ModuleList([NEXT_STAGE_G_SAME(ngf, use_hrc=1)])
        self.p_fg_net = nn.ModuleList([TO_RGB_LAYER(ngf // 2)])
        self.p_mk_net = nn.ModuleList([TO_GRAY_LAYER(ngf // 2)])

        ngf = ngf // 2

        for _ in range(start_depth):
            self.b_code_net.append(NEXT_STAGE_G_UP(ngf, use_hrc=0))
            self.b_img_net.append(TO_RGB_LAYER(ngf // 2))

            self.p_code_net.append(NEXT_STAGE_G_UP(ngf, use_hrc=1))
            self.p_fg_net.append(TO_RGB_LAYER(ngf // 2))
            self.p_mk_net.append(TO_GRAY_LAYER(ngf // 2))

            ngf = ngf // 2

        self.gf_dim = ngf
        self.cur_depth = start_depth

    def inc_depth(self):
        # oneline apply should work??
        ngf = self.gf_dim

        self.b_code_net.append(NEXT_STAGE_G_UP(
            ngf, use_hrc=0).apply(weights_init))
        self.b_img_net.append(TO_RGB_LAYER(ngf // 2).apply(weights_init))

        self.p_code_net.append(NEXT_STAGE_G_UP(
            ngf, use_hrc=1).apply(weights_init))

        self.p_fg_net.append(TO_RGB_LAYER(ngf // 2).apply(weights_init))
        self.p_mk_net.append(TO_GRAY_LAYER(ngf // 2).apply(weights_init))

        ngf = ngf // 2
        self.gf_dim = ngf
        self.cur_depth += 1

    def forward(self, z_code, p_code, b_code, alpha=None):
        raw_imgs = []  # raw background and foreground
        fake_img = []  # Will contain [parent foreground, child foreground]
        mk_img = []  # Will contain [parent mask, child mask]

        # bsz = z_code.size(0)
        # pid = torch.randint(0, cfg.FG_CATEGORIES, (bsz,))
        # bid = torch.randint(0, cfg.BG_CATEGORIES, (bsz,))
        # p_code = torch.zeros([bsz, cfg.FG_CATEGORIES]).cuda()
        # b_code = torch.zeros([bsz, cfg.BG_CATEGORIES]).cuda()
        # for i in range(bsz):
        #     p_code[i, pid[i]] = 1
        #     b_code[i, bid[i]] = 1

        h_code_b = self.b_code_init(z_code, b_code)
        h_code_p = self.p_code_init(z_code, p_code)

        for i in range(self.cur_depth + 1):
            _b_code_net = self.b_code_net[i]
            _b_img_net = self.b_img_net[i]
            _p_code_net = self.p_code_net[i]
            _p_fg_net = self.p_fg_net[i]
            _p_mk_net = self.p_mk_net[i]

            h_code_b = _b_code_net(h_code_b, b_code)

            fake_img1 = _b_img_net(h_code_b)
            if i == self.cur_depth and i != start_depth and alpha < 1:
                prev_fake_img1 = raw_imgs[(i-1)*3]
                prev_fake_img1 = F.upsample(
                    prev_fake_img1, scale_factor=2)  # mode='nearest'
                fake_img1 = (1 - alpha) * prev_fake_img1 + alpha * fake_img1

            h_code_p = _p_code_net(h_code_p, p_code)

            fake_img2 = _p_fg_net(h_code_p)  # Parent foreground
            fake_img2_mk = _p_mk_net(h_code_p)  # Parent mask
            if i == self.cur_depth and i != start_depth and alpha < 1:
                prev_fake_img2 = fake_img[(i-1)*2]
                prev_fake_img2 = F.upsample(
                    prev_fake_img2, scale_factor=2)  # mode='nearest'
                fake_img2 = (1 - alpha) * \
                    prev_fake_img2 + alpha * fake_img2
                prev_fake_img2_mk = mk_img[(i-1)*2]
                prev_fake_img2_mk = F.upsample(
                    prev_fake_img2_mk, scale_factor=2)  # mode='nearest'
                fake_img2_mk = (1 - alpha) * \
                    prev_fake_img2_mk + alpha * fake_img2_mk

            if i == self.cur_depth:
                ones_mask_p = torch.ones_like(fake_img2_mk)
                opp_mask_p = ones_mask_p - fake_img2_mk
                fg_masked2 = torch.mul(fake_img2, fake_img2_mk)
                bg_masked2 = torch.mul(fake_img1, opp_mask_p)
                fake_img2_final = fg_masked2 + bg_masked2  # Parent image

                raw_imgs.append(fake_img1)
                raw_imgs.append(fake_img2)
                fake_img.append(fake_img2_final)
                mk_img.append(fake_img2_mk)

        return fake_img, raw_imgs, mk_img

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


class D_NET_PC_BASE(nn.Module):
    def __init__(self, stg_no, ndf):
        super().__init__()
        self.df_dim = ndf
        self.stg_no = stg_no
        if self.stg_no == 1:
            self.ef_dim = cfg.FG_CATEGORIES
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
        self.jointConv = Block3x3_leakRelu(ndf * 2, ndf * 2)
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 2, efg, kernel_size=4, stride=4))
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

    def forward(self, x_var, alpha=None):
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



# mask predict

def Up_unet(in_c, out_c):
    return nn.Sequential(nn.ConvTranspose2d(in_c, out_c*2, 4, 2, 1), nn.BatchNorm2d(out_c*2), GLU())


# def BottleNeck(in_c, out_c):
#     return nn.Sequential(nn.Conv2d(in_c, out_c*2, 4, 4), nn.BatchNorm2d(out_c*2), GLU())


def Down_unet(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_c*2, 4, 2, 1), nn.BatchNorm2d(out_c*2), GLU())


class MaskPredictor(nn.Module):
    def __init__(self, in_c, out_c=1):
        super().__init__()

        self.first = nn.Sequential(sameBlock(in_c, 32), sameBlock(32, 32))

        self.down1 = Down_unet(32, 32)
        # 32*64*64
        self.down2 = Down_unet(32, 64)
        # 64*32*32
        self.down3 = Down_unet(64, 128)
        # 128*16*16
        self.down4 = Down_unet(128, 256)
        # 256*8*8
        self.down5 = Down_unet(256, 512)
        # 512*4*4
        self.down6 = Down_unet(512, 512)
        # 512*2*2

        self.up1 = Up_unet(512, 256)
        # 256*4*4
        self.up2 = Up_unet(256+512, 512)
        # 256*8*8
        self.up3 = Up_unet(512+256, 256)
        # 256*16*16
        self.up4 = Up_unet(256+128, 128)
        # 128*32*32
        self.up5 = Up_unet(128+64, 64)
        # 64*64*64
        self.up6 = Up_unet(64+32, out_c)
        # out_c*128*128

        self.last = nn.Sequential(
            ResBlock(out_c),
            # ResBlock(out_c),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.first(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(self.down6(x5))

        x = self.up2(torch.cat([x, x5], dim=1))
        x = self.up3(torch.cat([x, x4], dim=1))
        x = self.up4(torch.cat([x, x3], dim=1))
        x = self.up5(torch.cat([x, x2], dim=1))
        x = self.up6(torch.cat([x, x1], dim=1))

        return self.last(x)


## code img pair discriminator
class Gaussian(nn.Module):
    def __init__(self, std):
        super(Gaussian, self).__init__()
        self.std = std

    def forward(self, x):
        n = torch.randn_like(x)*self.std
        return x+n

class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c, k, s, p=0, bn=True, activation=None, noise=False, std=None):
        super(Conv_Block, self).__init__()
        model = [nn.Conv2d(in_c, out_c, k, s, p)]

        if bn:
            model.append(nn.BatchNorm2d(out_c))

        if activation == 'relu':
            model.append(nn.ReLU())
        if activation == 'elu':
            model.append(nn.ELU())
        if activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        if activation == 'tanh':
            model.append(nn.Tanh())
        if activation == 'sigmoid':
            model.append(nn.Sigmoid())
        if activation == 'softmax':
            model.append(nn.Softmax(dim=1))

        if noise:
            model.append(Gaussian(std))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Linear_Block(nn.Module):
    def __init__(self, in_c, out_c, bn=True, activation=None, noise=False, std=None):
        super(Linear_Block, self).__init__()
        model = [nn.Linear(in_c, out_c)]

        if bn:
            model.append(nn.BatchNorm1d(out_c))

        if activation == 'relu':
            model.append(nn.ReLU())
        if activation == 'elu':
            model.append(nn.ELU())
        if activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        if activation == 'tanh':
            model.append(nn.Tanh())
        if activation == 'sigmoid':
            model.append(nn.Sigmoid())
        if activation == 'softmax':
            model.append(nn.Softmax(dim=1))

        if noise:
            model.append(Gaussian(std))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Viewer(nn.Module):
    def __init__(self, shape):
        super(Viewer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Bi_Dis_base(nn.Module):
    def __init__(self, code_len, ngf=16):
        super(Bi_Dis_base, self).__init__()

        self.image_layer = nn.Sequential(  Conv_Block( 3,     ngf, 4,2,1, bn=False, activation='leaky', noise=False, std=0.3),
                                             Conv_Block( ngf,   ngf*2, 4,2,1, bn=False, activation='leaky', noise=False, std=0.5),
                                             Conv_Block( ngf*2, ngf*4, 4,2,1, bn=False, activation='leaky', noise=False, std=0.5),
                                             Conv_Block( ngf*4, ngf*8, 4,2,1, bn=False, activation='leaky', noise=False, std=0.5),
                                             Conv_Block( ngf*8, ngf*16, 4,2,1, bn=False, activation='leaky', noise=False, std=0.5),
                                             Conv_Block( ngf*16, 512, 4,1,0, bn=False, activation='leaky', noise=False, std=0.5),
                                             Viewer( [-1,512] ) )

        self.code_layer = nn.Sequential( Linear_Block( code_len, 512, bn=False, activation='leaky', noise=True, std=0.5  ),
                                           Linear_Block( 512, 512, bn=False, activation='leaky', noise=True, std=0.3  ),
                                           Linear_Block( 512, 512, bn=False, activation='leaky', noise=True, std=0.3  ) )

        self.joint = nn.Sequential(  Linear_Block(1024,1024, bn=False, activation='leaky', noise=False, std=0.5),
                                     Linear_Block(1024, 1,  bn=False,  activation='None' ),
                                     nn.Sigmoid(),
                                     Viewer([-1]) )

    def forward(self, img, code ):
        t1 = self.image_layer(img)
        t2 = self.code_layer( code )
        return  self.joint(  torch.cat( [t1,t2],dim=1) )



class Bi_Dis(nn.Module):
    def __init__(self):
        super(Bi_Dis, self).__init__()
        self.BD_b = Bi_Dis_base(cfg.BG_CATEGORIES)
        self.BD_p = Bi_Dis_base(cfg.FG_CATEGORIES)

    def forward(self, img, b_code, p_code, mask=None):
        mask = None
        if mask is None:
            which_pair_b = self.BD_b(img, b_code)
            which_pair_p = self.BD_p(img, p_code)
        else:
            bg = (torch.ones_like(mask) - mask) * img
            fg = mask * img
            which_pair_b = self.BD_b(bg, b_code)
            which_pair_p = self.BD_p(fg, p_code)

        return which_pair_b, which_pair_p
