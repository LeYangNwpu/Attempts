import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import SegNet_resnet_utils as uitls


class SkipConnection(nn.Module):

    def __init__(self, in_planes, out_planes, transition_up=False):
        super(SkipConnection, self).__init__()
        self.transition_up = transition_up
        self.relu = nn.ReLU(inplace=True)
        mid_in_planes = int(round(math.sqrt(in_planes * out_planes)))
        self.conv_skip1 = uitls.conv3x3(in_planes, mid_in_planes)
        self.bn_skip1 = nn.BatchNorm2d(mid_in_planes)
        self.conv_skip2 = uitls.conv3x3(mid_in_planes, out_planes)
        self.bn_skip2 = nn.BatchNorm2d(out_planes)
        self.conv_in = uitls.conv3x3(in_planes, out_planes)
        self.bn_in = nn.BatchNorm2d(out_planes)


    def forward(self, data):
        x_in = data[0]
        x_skip = data[1]
        if self.transition_up:
            # x_in = F.upsample(x_in, scale_factor=2, mode='bilinear', align_corners=True)
            x_in = F.upsample(x_in, scale_factor=2, mode='bilinear')

        # x_in center crop to facilitate the cat operation
        x_in = uitls.center_crop(x_in, x_skip.size(2), x_skip.size(3))
        fea_in = self.conv_in(x_in)
        fea_in = self.bn_in(fea_in)

        # x_skip
        x_skip = self.conv_skip1(x_skip)
        x_skip = self.bn_skip1(x_skip)
        x_skip = self.relu(x_skip)
        fea_skip = self.conv_skip2(x_skip)
        fea_skip = self.bn_skip2(fea_skip)

        # output
        out = fea_in + fea_skip
        out = self.relu(out)

        return out
