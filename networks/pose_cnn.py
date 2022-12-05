# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import timm


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class PoseResNet(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseResNet, self).__init__()

        self.num_input_frames = num_input_frames

        self.resnet = timm.create_model("resnet18", pretrained=True)
        self.resnet.fc = Identity()

        self.pose_conv = nn.Linear(
            512 * num_input_frames, 6 * (num_input_frames - 1), 1
        )

    def forward(self, out):

        B, N, C, H, W = out.shape
        out = self.resnet(out.reshape(-1, C, H, W))
        out = out.reshape(B, -1)

        out = self.pose_conv(out)
        out = out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class PoseViT(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseViT, self).__init__()

        self.num_input_frames = num_input_frames

        self.resnet = timm.create_model("mobilenetv2_100", pretrained=True)
        self.resnet.classifier = Identity()

        self.pose_conv = nn.Linear(
            1280 * num_input_frames, 6 * (num_input_frames - 1), 1
        )

    def forward(self, out):

        B, N, C, H, W = out.shape
        out = self.resnet(out.reshape(-1, C, H, W))
        out = out.reshape(B, -1)

        out = self.pose_conv(out)
        out = out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
