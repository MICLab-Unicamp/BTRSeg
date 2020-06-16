'''
My personal UNet code. Heavily modified from internet code.

Defines the UNet model and its parts/possible modifications
Contains options to use/not use modifications described on the paper
(residual connections, bias, batch_norm etc)

Author: Diedre Carmo
https://github.com/dscarmo
'''
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.device import get_device


GLOBAL_GROUP = 8


def assert_dim(dim):
    assert dim in ('2d', '3d'), "dim {} not supported".format(dim)


def conv(in_ch, out_ch, kernel_size, padding, bias, dim='2d', stride=1):
    assert_dim(dim)
    if dim == '2d':
        return nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=bias, stride=stride)
    elif dim == '3d':
        return nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, bias=bias, stride=stride)


def batch_norm(out_ch, dim='2d'):
    assert_dim(dim)
    if dim == '2d':
        return nn.BatchNorm2d(out_ch)
    elif dim == '3d':
        return nn.BatchNorm3d(out_ch)


def max_pool(kernel_size, dim='2d'):
    assert_dim(dim)
    if dim == '2d':
        return nn.MaxPool2d(kernel_size)
    elif dim == '3d':
        return nn.MaxPool3d(kernel_size)


def conv_transpose(in_ch, out_ch, kernel_size, stride, bias, dim='2d'):
    assert_dim(dim)
    if dim == '2d':
        return nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size, stride=stride, bias=bias)
    elif dim == '3d':
        return nn.ConvTranspose3d(in_ch//2, in_ch//2, kernel_size, stride=stride, bias=bias)


class self_attention(nn.Module):
    '''
    Spatial attention module, with 1x1 convolutions, idea from
    ASSESSING KNEE OA SEVERITY WITH CNN ATTENTION-BASED END-TO-END ARCHITECTURES
    '''
    def __init__(self, in_ch, bias=False, dim='2d', open_the_gates=False):
        super().__init__()
        if not open_the_gates:
            self.first_conv = conv(in_ch, in_ch//2, 1, 0, bias, dim=dim)
            self.second_conv = conv(in_ch//2, in_ch//4, 1, 0, bias, dim=dim)
            self.third_conv = conv(in_ch//4, 1, 1, 0, bias, dim=dim)
        self.open_the_gates = open_the_gates

    def forward(self, x):
        if self.open_the_gates:
            return x
        else:
            y = self.first_conv(x)
            y = F.leaky_relu(y, inplace=True)
            y = self.second_conv(y)
            y = F.leaky_relu(y, inplace=True)
            self.att = self.third_conv(y).sigmoid()
            return x*self.att


class double_conv(nn.Module):
    '''
    (conv => BN => ReLU) * 2, one UNET Block
    '''
    def __init__(self, in_ch, out_ch, residual=False, bias=False, bn=True, dim='2d'):
        super(double_conv, self).__init__()
        global GLOBAL_GROUP
        if bn == "group":
            norm1 = nn.GroupNorm(num_groups=GLOBAL_GROUP, num_channels=out_ch)
            norm2 = nn.GroupNorm(num_groups=GLOBAL_GROUP, num_channels=out_ch)
        elif bn:
            norm1 = batch_norm(out_ch, dim=dim)
            norm2 = batch_norm(out_ch, dim=dim)
        else:
            norm1 = nn.Identity()
            norm2 = nn.Identity()

        self.conv = nn.Sequential(
            conv(in_ch, out_ch, 3, 1, bias, dim=dim),
            norm1,
            nn.LeakyReLU(inplace=True),
            conv(out_ch, out_ch, 3, 1, bias, dim=dim),
            norm2,
            nn.LeakyReLU(inplace=True)
        )

        self.residual = residual
        if residual:
            self.residual_connection = conv(in_ch, out_ch, 1, 0, bias, dim=dim)

    def forward(self, x):
        y = self.conv(x)

        if self.residual:
            return y + self.residual_connection(x)
        else:
            return y


class inconv(nn.Module):
    '''
    Input convolution
    '''
    def __init__(self, in_ch, out_ch, residual=False, bias=False, bn=True, dim='2d'):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, residual=residual, bias=bias, bn=bn, dim=dim)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    '''
    Downsample conv
    '''
    def __init__(self, in_ch, out_ch, residual=False, bias=False, bn=True, dim='2d'):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            max_pool(2, dim=dim),
            double_conv(in_ch, out_ch, residual=residual, bias=bias, bn=bn, dim=dim)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    '''
    Upsample conv
    '''
    def __init__(self, in_ch, out_ch, residual=False, bias=False, bn=True, dim='2d'):
        super(up, self).__init__()
        self.up = conv_transpose(in_ch, in_ch, 2, 2, bias=bias, dim=dim)
        self.conv = double_conv(in_ch, out_ch, residual, bias=bias, bn=bn, dim=dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    '''
    Output convolution
    '''
    def __init__(self, in_ch, out_ch, bias=False, dim='2d'):
        super(outconv, self).__init__()
        self.conv = conv(in_ch, out_ch, 1, 0, bias, dim=dim)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    '''
    Main model class
    '''
    @staticmethod
    def forward_test(inpt, n_channels, n_classes, apply_sigmoid, apply_softmax, residual, small, bias, bn, attention=True,
                     dim='2d', limit=10, channel_factor=4):
        '''
        UNet unit test
        '''
        unet = UNet(n_channels, n_classes, apply_sigmoid=apply_sigmoid, residual=residual, small=small, bias=bias, bn=bn, dim=dim,
                    use_attention=attention, apply_softmax=apply_softmax,
                    channel_factor=channel_factor).to(get_device())
        print(unet)
        for _ in tqdm(range(limit), desc=f"Test forwarding {limit} times."):
            _ = unet(inpt)

        return unet(inpt)

    def __init__(self, n_channels, n_classes, apply_sigmoid=True, residual=True, small=False, bias=False, bn=True, verbose=True,
                 dim='2d', use_attention=False, apply_softmax=False, channel_factor=1):
        super(UNet, self).__init__()
        assert bn in [True, "group", False]
        global GLOBAL_GROUP

        if channel_factor == 16:
            GLOBAL_GROUP = 4
        elif channel_factor == 32:
            GLOBAL_GROUP = 2

        big = not small
        self.inc = inconv(n_channels, 64//channel_factor, residual, bias, bn, dim=dim)
        self.down1 = down(64//channel_factor, 128//channel_factor, residual, bias, bn, dim=dim)
        self.down2 = down(128//channel_factor, (128+big*128)//channel_factor, residual, bias, bn, dim=dim)
        if not small:
            self.down3 = down(256//channel_factor, 512//channel_factor, residual, bias, bn, dim=dim)
            self.down4 = down(512//channel_factor, 512//channel_factor, residual, bias, bn, dim=dim)
            self.up1 = up(1024//channel_factor, 256//channel_factor, residual, bias, bn, dim=dim)
            self.up2 = up(512//channel_factor, 128//channel_factor, residual, bias, bn, dim=dim)
        self.up3 = up(256//channel_factor, 64//channel_factor, residual, bias, bn, dim=dim)
        self.up4 = up(128//channel_factor, 64//channel_factor, residual, bias, bn, dim=dim)
        self.outc = outconv(64//channel_factor, n_classes, bias, dim=dim)

        assert not(apply_sigmoid and apply_softmax), "apply sigmoid OR softmax, not both!"

        self.apply_sigmoid = apply_sigmoid
        self.apply_softmax = apply_softmax

        self.small = small

        self.x1_gate = self_attention(64//channel_factor, bias=bias, dim=dim, open_the_gates=not use_attention)
        self.x2_gate = self_attention(128//channel_factor, bias=bias, dim=dim, open_the_gates=not use_attention)
        self.x3_gate = self_attention(256//channel_factor, bias=bias, dim=dim, open_the_gates=not use_attention)
        self.x4_gate = self_attention(512//channel_factor, bias=bias, dim=dim, open_the_gates=not use_attention)

        self.use_attention = use_attention

        if verbose:
            print(f"UNet using sigmoid: {apply_sigmoid} residual connections: {residual} small: {small} in channels: {n_channels}"
                  f" bias: {bias} batch_norm: {bn} dim:"
                  f" {dim} attention: {use_attention} apply_softmax: {apply_softmax} out_channels {n_classes}")

    def forward(self, x):
        '''
        Saves every downstep output to use in upsteps concatenation
        '''
        if self.small:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up3(x3, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.down4(x4)

            out_up1 = self.up1(x, self.x4_gate(x4))
            out_up2 = self.up2(out_up1, self.x3_gate(x3))
            out_up3 = self.up3(out_up2, self.x2_gate(x2))
            x = self.up4(out_up3, self.x1_gate(x1))
            x = self.outc(x)

        if self.apply_sigmoid:
            x = x.sigmoid()
        elif self.apply_softmax:
            x = x.softmax(dim=1)  # expects [B, C, ...] shape

        return x


def init_weights(vgg, model, verbose=False):
    '''
    Returns updated state dict to be loaded into model pre training with vgg weights
    '''
    assert model.dim == '2d'
    state_dict = model.state_dict()
    for vk, vv in vgg.state_dict().items():
        if vk.split('.')[-1] == "weight" and vk.split('.')[0] != "classifier":
            for uk, uv in state_dict.items():
                if uk.split('.')[-1] == "weight" and uk.split('.')[0][:2] != "up":
                    if vv.shape == uv.shape:
                        if verbose:
                            print("Found compatible layer...")
                            print("VGG Key: {}".format(vk))
                            print("UNET Key: {}".format(uk))
                            print("VGG shape: {}".format(vv.shape))
                            print("UNET shape: {}".format(uv.shape))
                        state_dict[uk] = vv
                        if verbose:
                            print("Weights transfered. Check:", end=' ')
                            # one liner to check if all weights are the same
                            print(((state_dict[uk] == vgg.state_dict()[vk]).sum()/len(state_dict[uk].view(-1))).item() == 1)
                            print("-"*20)
                        break
    print("VGG11 weigths transfered to compatible unet encoder layers.")
    return state_dict


def test_unet(display=False, long_test=False):
    test_input = torch.randn((2, 4, 32, 32, 32)).cuda()
    out = UNet.forward_test(test_input, 4, 3, apply_sigmoid=False, apply_softmax=False,
                            residual=True, small=False, bias=False, bn="group", attention=True,
                            dim='3d', limit=10)
    print(test_input.shape, out.shape)
