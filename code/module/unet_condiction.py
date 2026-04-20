# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.time_mlp = torch.nn.Linear(512, out_channels)

    def forward(self, x, temp):
        # print(temp.shape,x.shape)
        # temb = self.time_mlp(nonlinearity(temp))[:, :, None, None]
        x = self.conv_conv(x)
        # print(x.shape)
        temb = self.time_mlp(nonlinearity(temp))[:, :, None, None]
        # print(temb.shape,x.shape)
        return x+temb


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     ConvBlock(in_channels, out_channels, dropout_p)

        # )
        self.maxpool = nn.MaxPool2d(2)
        self.conv_block = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x, temp):
        x = self.maxpool(x)
        return self.conv_block(x,temp)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2,temp):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x,temp)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        # self.temb_proj = torch.nn.Linear(512, self.in_chns)

        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x, temp):
        # temb = self.temb_proj(nonlinearity(temp))[:, :, None, None]
        x0 = self.in_conv(x,temp)
        x1 = self.down1(x0,temp)
        x2 = self.down2(x1,temp)
        x3 = self.down3(x2,temp)
        x4 = self.down4(x3,temp)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)
        # self.temb_proj = torch.nn.Linear(512, self.in_chns)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature, temp):
        # temb = self.temb_proj(nonlinearity(temp))[:, :, None, None]
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        y4 = self.up1(x4, x3,temp)
        y3 = self.up2(y4, x2,temp)
        y2 = self.up3(y3, x1,temp)
        y1 = self.up4(y2, x0,temp)
        output = self.out_conv(y1)
        return output


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x

import math
def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb
def nonlinearity(x):
    return x * torch.sigmoid(x)


class UNet_project(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_project, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.Wx = nn.Parameter(torch.randn(512, 14), requires_grad=True)
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128, 512),
            torch.nn.Linear(512, 512),
        ])

    def forward(self, x: torch.Tensor, t, image):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        x = torch.cat([x, image], dim=1)
        feature = self.encoder(x,temb)
        output = self.decoder(feature,temb)
        if isinstance(feature, list):
            feature = feature[4]

        # 线性投影操作
        projected = torch.matmul(self.Wx, feature.transpose(1, 0)).transpose(1, 0)
        projected_normalized = torch.nn.functional.normalize(projected, p=2, dim=1)

        # # 线性投影操作
        # projected = torch.matmul(self.Wx, feature.transpose(1, 0)).transpose(1, 0)  # 投影到 Dh 维
        # # 在单位超球面上进行归一化
        # projected_normalized = F.normalize(projected, p=2, dim=1)  # 归一化到 (batch_size, Dh)
        return output,projected_normalized


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        # self.Wx = nn.Parameter(torch.randn(512, 14), requires_grad=True)
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128, 512),
            torch.nn.Linear(512, 512),
        ])

    def forward(self, x: torch.Tensor, t, image):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        x = torch.cat([x, image], dim=1)
        feature = self.encoder(x,temb)
        output = self.decoder(feature,temb)
        return output
    

    
    
class UNet_Anchor(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_Anchor, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.Wx = nn.Parameter(torch.randn(512, 14), requires_grad=True)
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128, 512),
            torch.nn.Linear(512, 512),
        ])
        self.conv1x1 = nn.Conv2d(in_chns, in_chns, kernel_size=1)
        # self.conv1x1 = nn.Conv2d(in_chns+1, in_chns, kernel_size=1)
        # self.conv1x1 = nn.Conv2d(1, 1, kernel_size=1)
        # 可学习的高斯平滑核
        # self.gaussian_kernel = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.gaussian_kernel = nn.Conv2d(in_chns, in_chns, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor, t, image):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        smoothed_Fc = self.gaussian_kernel(x)
        fanc = torch.max(smoothed_Fc,x)

        anchor_feature = torch.sigmoid(self.conv1x1(fanc))*image
        x = anchor_feature+image
        # print(x.shape)
        # smoothed_Fc = self.gaussian_kernel(image)
        # fanc = torch.max(smoothed_Fc,image)

        # anchor_feature = torch.sigmoid(self.conv1x1(fanc))*x
        # x = anchor_feature+x

        # smoothed_Fc = self.gaussian_kernel(x)
        # # print(smoothed_Fc.shape)
        # anchor_Feat = self.conv1x1(torch.cat([smoothed_Fc,image],dim=1))
        # # fanc = torch.max(smoothed_Fc,image)

        # anchor_feature = torch.sigmoid(anchor_Feat)*image
        # x = anchor_feature+image

        # x = torch.cat([x, image], dim=1)
        feature = self.encoder(x,temb)
        output = self.decoder(feature,temb)
        return output