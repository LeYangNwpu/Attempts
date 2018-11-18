import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind

import BasicBlock
import Bottleneck
import SkipConnection
import SegNet_resnet_utils as uitls


class ResNet(nn.Module):

    def __init__(self, block, layers, skip_connection, deep_mask=None):
        super(ResNet, self).__init__()
        self.deep_mask = deep_mask
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # remove the pooling operation, use dilation convolution
        self.layer3 = self._make_layer(block, 256, layers[2], dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], dilation=4)

        # bottleneck
        self.bottleneck = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)

        # skip connection and transition convolution
        self.skip3 = self._make_connection(skip_connection, in_planes=256, out_planes=128)
        self.skip2 = self._make_connection(skip_connection, in_planes=128, out_planes=64)
        self.skip1 = self._make_connection(skip_connection, in_planes=64, out_planes=64, transition_up=True)
        self.skip0 = self._make_connection(skip_connection, in_planes=64, out_planes=32, transition_up=True)

        # self.LSoftmax = nn.LogSoftmax(dim=1)
        self.LSoftmax = nn.LogSoftmax()
        self.bn_fea = nn.BatchNorm2d(36)

        self.mask0_conv = uitls.conv1x1(32, 36)
        self.mask0 = uitls.conv3x3(36, 2)
        self.seg2map = uitls.conv1x1(10, 15)
        self.map2mask = uitls.conv1x1(15, 2)
        if deep_mask:
            self.mask4_conv = uitls.conv1x1(256, 36)
            self.mask4 = uitls.conv3x3(36, 2)
            self.mask3_conv = uitls.conv1x1(128, 36)
            self.mask3 = uitls.conv3x3(36, 2)
            self.mask2_conv = uitls.conv1x1(64, 36)
            self.mask2 = uitls.conv3x3(36, 2)
            self.mask1_conv = uitls.conv1x1(64, 36)
            self.mask1 = uitls.conv3x3(36, 2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_connection(self, skip_connection, in_planes, out_planes, transition_up=False):
        layers = []
        layers.append(skip_connection(in_planes, out_planes, transition_up))

        return nn.Sequential(*layers)

    def fuse_feature(self, features):
        '''
        fuse 5 feature cubes into one feature (180x112x112)
        '''
        data = features[0]
        bat, cha, hei, wid = data.size()
        num_fea = len(features)
        fea = torch.zeros((bat, cha*num_fea, hei, wid))
        fea = fea.type(torch.FloatTensor)
        fea = Variable(fea).cuda()
        for ibat in range(bat):
            for icha in range(cha):
                for jfea in range(num_fea):
                    cha_c = jfea + icha * num_fea
                    cube = features[jfea]
                    fea[ibat, cha_c, :, :] = cube[ibat, icha, :, :]
        return fea

    def fuse_seg(self, segs):
        '''
        fuse 5 segmentation masks into one feature (N * 10 * 112 * 112)
        '''
        data = segs[0]
        bat, cha, hei, wid = data.size()
        assert cha == 2
        num_mask = len(segs)
        mask_fore = torch.zeros((bat, num_mask, hei, wid))
        mask_fore = mask_fore.type(torch.FloatTensor)
        mask_fore = Variable(mask_fore).cuda()
        mask_back = torch.zeros((bat, num_mask, hei, wid))
        mask_back = mask_back.type(torch.FloatTensor)
        mask_back = Variable(mask_back).cuda()

        for ibat in range(bat):
            for jfea in range(num_mask):
                cube = segs[jfea]
                mask_fore[ibat, jfea, :, :] = cube[ibat, 0, :, :]
                mask_back[ibat, jfea, :, :] = cube[ibat, 1, :, :]
        mask = torch.cat((mask_fore, mask_back), dim=1)

        return mask

    def forward(self, x):
        skip_connect = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 112
        skip_connect.append(x)
        # 56
        x = self.maxpool(x)

        # 56
        x = self.layer1(x)
        skip_connect.append(x)
        # 28
        x = self.layer2(x)
        skip_connect.append(x)
        x = self.layer3(x)
        skip_connect.append(x)
        x = self.layer4(x)

        # bottleneck
        fea_map = self.bottleneck(x)
        mask4 = self.mask4_conv(fea_map)
        mask4 = self.bn_fea(mask4)
        # generate segmentation mask
        seg4 = self.relu(mask4)
        seg4 = self.mask4(seg4)
        seg4 = F.upsample(seg4, scale_factor=4, mode='bilinear')
        # mask4 = F.upsample(mask4, scale_factor=4, mode='bilinear', align_corners=True)
        mask4 = F.upsample(mask4, scale_factor=4, mode='bilinear')

        skip = skip_connect.pop()
        # 28
        x = self.skip3((fea_map, skip))
        mask3 = self.mask3_conv(x)
        mask3 = self.bn_fea(mask3)
        # generate segmentation mask
        seg3 = self.relu(mask3)
        seg3 = self.mask3(seg3)
        seg3 = F.upsample(seg3, scale_factor=4, mode='bilinear')
        mask3 = F.upsample(mask3, scale_factor=4, mode='bilinear')

        skip = skip_connect.pop()
        # 28
        x = self.skip2((x, skip))
        mask2 = self.mask2_conv(x)
        mask2 = self.bn_fea(mask2)
        # generate segmentation mask
        seg2 = self.relu(mask2)
        seg2 = self.mask2(seg2)
        seg2 = F.upsample(seg2, scale_factor=4, mode='bilinear')
        mask2 = F.upsample(mask2, scale_factor=4, mode='bilinear')

        skip = skip_connect.pop()
        # 56
        x = self.skip1((x, skip))
        mask1 = self.mask1_conv(x)
        mask1 = self.bn_fea(mask1)
        # generate segmentation mask
        seg1 = self.relu(mask1)
        seg1 = self.mask1(seg1)
        seg1 = F.upsample(seg1, scale_factor=2, mode='bilinear')
        mask1 = F.upsample(mask1, scale_factor=2, mode='bilinear')

        skip = skip_connect.pop()
        # 112
        x = self.skip0((x, skip))
        mask0 = self.mask0_conv(x)
        mask0 = self.bn_fea(mask0)
        # generate segmentation mask
        seg0 = self.relu(mask0)
        seg0 = self.mask0(seg0)

        # cat 5 segmentation masks
        # maybe, we should combine these segs in hand
        # segall = torch.cat((seg0, seg1, seg2, seg3, seg4), dim=1)
        segall = self.fuse_seg((seg4, seg3, seg2, seg1, seg0))
        segall = self.seg2map(segall)

        # feature map size is [2, 180, 112, 112]
        feature = self.fuse_feature((mask4, mask3, mask2, mask1, mask0))
        feature = self.relu(feature)
        seg_mask = self.spnNet(feature, segall)

        seg_mask = self.map2mask(seg_mask)
        seg_mask = self.LSoftmax(seg_mask)

        return seg_mask


    def spnNet(self, features, mask):
        '''
        spn refine the segmentation mask
        features:[N, 180, 112, 112]
        mask:[N, 15, 112, 112]
        '''

        # left->right:
        Propagator = GateRecurrent2dnoind(True, False)

        G1 = features[:, 0:15, :, :]
        G2 = features[:, 15:30, :, :]
        G3 = features[:, 30:45, :, :]

        sum_abs = G1.abs() + G2.abs() + G3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        G1_norm = torch.div(G1, sum_abs)
        G2_norm = torch.div(G2, sum_abs)
        G3_norm = torch.div(G3, sum_abs)

        G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
        G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
        G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm
        G1[G1 != G1] = 0.001
        G2[G2 != G2] = 0.001
        G3[G3 != G3] = 0.001

        mask_l2r = Propagator.forward(mask, G1, G2, G3)

        # right->left:
        Propagator_r2l = GateRecurrent2dnoind(True, True)

        G1 = features[:, 45:60, :, :]
        G2 = features[:, 60:75, :, :]
        G3 = features[:, 75:90, :, :]

        sum_abs = G1.abs() + G2.abs() + G3.abs()

        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()

        G1_norm = torch.div(G1, sum_abs)
        G2_norm = torch.div(G2, sum_abs)
        G3_norm = torch.div(G3, sum_abs)

        G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
        G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
        G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm
        G1[G1 != G1] = 0.001
        G2[G2 != G2] = 0.001
        G3[G3 != G3] = 0.001

        mask_r2l = Propagator_r2l.forward(mask, G1, G2, G3)

        # top->bottom:
        Propagator = GateRecurrent2dnoind(False, False)

        G1 = features[:, 90:105, :, :]
        G2 = features[:, 105:120, :, :]
        G3 = features[:, 120:135, :, :]

        sum_abs = G1.abs() + G2.abs() + G3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        G1_norm = torch.div(G1, sum_abs)
        G2_norm = torch.div(G2, sum_abs)
        G3_norm = torch.div(G3, sum_abs)

        G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
        G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
        G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm
        G1[G1 != G1] = 0.001
        G2[G2 != G2] = 0.001
        G3[G3 != G3] = 0.001

        mask_t2b = Propagator.forward(mask, G1, G2, G3)

        # bottom->top:
        Propagator = GateRecurrent2dnoind(False, True)

        G1 = features[:, 135:150, :, :]
        G2 = features[:, 150:165, :, :]
        G3 = features[:, 165:180, :, :]

        sum_abs = G1.abs() + G2.abs() + G3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        G1_norm = torch.div(G1, sum_abs)
        G2_norm = torch.div(G2, sum_abs)
        G3_norm = torch.div(G3, sum_abs)

        G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
        G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
        G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm
        G1[G1 != G1] = 0.001
        G2[G2 != G2] = 0.001
        G3[G3 != G3] = 0.001

        mask_b2t = Propagator.forward(mask, G1, G2, G3)

        # max
        mask1 = torch.max(mask_l2r, mask_r2l)
        mask2 = torch.max(mask_t2b, mask_b2t)
        result = torch.max(mask1, mask2)

        return result


def resnet18():
    model = ResNet(BasicBlock.BasicBlock, [2, 2, 2, 2], SkipConnection.SkipConnection)
    return model


def resnet34():
    model = ResNet(BasicBlock.BasicBlock, [3, 4, 6, 3], SkipConnection.SkipConnection, deep_mask=True)
    return model


def resnet50():
    model = ResNet(Bottleneck.Bottleneck, [3, 4, 6, 3], SkipConnection.SkipConnection, deep_mask=True)
    return model


def resnet101():
    model = ResNet(Bottleneck.Bottleneck, [3, 4, 23, 3], SkipConnection.SkipConnection)
    return model

# for i in range(100):
#     image = torch.randn((2, 3, 224, 224))
#     image = Variable(image).cuda()
#     model = resnet34().cuda()
#     result = model(image)
#     print(result.size())
#     mask = result.data.cpu()
#     # print(mask[0,0,:10,:10])
#     print(mask.min())
#     print(mask.max())
