# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from re import S
from this import s
from tkinter import E
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import cv2

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .ResNetV2 import ResNetV2
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .GatedSpatialConv import GatedSpatialConv2d as GSC
from .Resnet import BasicBlock as Block
from .RCAB import CALayer

BatchNorm = SynchronizedBatchNorm2d
logger = logging.getLogger(__name__)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class shape(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = Block(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = Block(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = Block(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn7 = nn.Conv2d(1024, 1, 1)
        self.fuse = nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gate1 = GSC(32, 32)
        self.gate2 = GSC(16, 16)
        self.gate3 = GSC(8, 8)

    def forward(self, x, features):
        x_size = x.size()

        s3 = F.interpolate(self.dsn3(features[2]), x_size[2:],
                           mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(features[1]), x_size[2:],
                           mode='bilinear', align_corners=True)

        im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()

        cs = self.res1(features[3])
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d1(cs)
        cs = self.gate1(cs, s3)
        cs = self.res2(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s4)

        cs = self.fuse(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(cs)
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)

        return edge_out, acts


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class UP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_less = Conv2dReLU(
            1024,
            512,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv11 = Conv2dReLU(1024, 256)
        self.conv21 = Conv2dReLU(256, 256)
        self.conv12 = Conv2dReLU(512, 64)
        self.conv22 = Conv2dReLU(64, 64)
        self.conv13 = Conv2dReLU(128, 64)
        self.conv23 = Conv2dReLU(64, 64)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, features):
        x0 = self.conv_less(features[0])
        x0 = self.up(x0)
        x1 = torch.cat((x0, features[1]), dim=1)
        x1 = self.conv11(x1)
        x1 = self.conv21(x1)
        x1 = self.up(x1)

        x2 = torch.cat((x1, features[2]), dim=1)
        x2 = self.conv12(x2)
        x2 = self.conv22(x2)
        x2 = self.up(x2)

        x3 = torch.cat((x2, features[3]), dim=1)
        x3 = self.conv13(x3)
        x3 = self.conv23(x3)
        x = self.up(x3)

        #x = self.conv14(x)
        #x = self.conv24(x)

        return x


class Down(nn.Module):
    def __init__(self, config):
        super(Down, self).__init__()
        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        features = self.hybrid_model(x)
        return features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class SSUNet(nn.Module):
    def __init__(self, config, img_size=512, num_classes=1, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.Down = Down(config)
        self.decoder = UP(config)

        self.segmentation_head = SegmentationHead(
            in_channels=128,
            out_channels=config['n_classes'],
            kernel_size=3,
        )

        self.config = config
        self.conv = nn.Conv2d(1, 64, 1)
        self.shape = shape()
        self.CAL = CALayer(128)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.Down(x)  # (B, n_patch, hidden)
        edge, acts = self.shape(x, features)
        edges = self.conv(acts)
        x = self.decoder(features)
        seg = torch.cat((x, edges), dim=1)
        seg = self.CAL(seg)

        logits = self.segmentation_head(seg)
        return logits, edge


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),  #
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
