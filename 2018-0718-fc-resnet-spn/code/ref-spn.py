import torch
import torch.nn as nn
from pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def dila_conv3x3(in_planes, out_planes, dilation):
    return  nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3,
                      dilation=dilation, padding=dilation, bias=False)

def center_crop(layer, max_height, max_width):
    #https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py#L162
    #Author does a center crop which crops both inputs (skip and upsample) to size of minimum dimension on both w/h
    batch_size, n_channels, layer_height, layer_width = layer.size()
    xy1 = (layer_width - max_width) // 2
    xy2 = (layer_height - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


class SkipConnection(nn.Module):

    def __init__(self, in_planes, skip_planes, out_planes_half, transition_up=False):
        super(SkipConnection, self).__init__()
        self.conv_in = conv3x3(in_planes, out_planes_half)
        self.bn = nn.BatchNorm2d(out_planes_half)
        self.relu = nn.ReLU(inplace=True)
        self.conv_skip = conv3x3(skip_planes, out_planes_half)
        self.transition_up = transition_up
        self.conv_cat = conv3x3(2*out_planes_half, 2*out_planes_half)
        self.bn_cat = nn.BatchNorm2d(2*out_planes_half)
        self.conv_trans = nn.ConvTranspose2d(in_channels=2*out_planes_half,
                                             out_channels=2*out_planes_half, kernel_size=3, stride=2)

    def forward(self, data):
        x_in = data[0]
        x_skip = data[1]
        x_in = self.conv_in(x_in)
        x_in = self.bn(x_in)
        x_in = self.relu(x_in)

        x_skip = self.conv_skip(x_skip)
        x_skip = self.bn(x_skip)
        x_skip = self.relu(x_skip)

        # center crop x_in to facilitate the cat operation
        x_in = center_crop(x_in, x_skip.size(2), x_skip.size(3))
        out = torch.cat((x_in, x_skip), dim=1)
        out = self.conv_cat(out)
        out = self.bn_cat(out)
        out = self.relu(out)

        if self.transition_up:
            out = self.conv_trans(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.dial_conv1 = dila_conv3x3(inplanes, planes, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.dial_conv2 = dila_conv3x3(planes, planes, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        if self.dilation is not None:
            out = self.dial_conv1(x)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dilation is not None:
            out = self.dial_conv2(out)
        else:
            out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, skip_connection, batchsize=1):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1_ch = conv3x3(3, 64, stride=1)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # remove the pooling operation, use dilation convolution
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # bottleneck
        # self.map_to_vect = nn.Linear(batchsize*512*7*7, 512)
        # self.vect_to_map = nn.Linear(512, batchsize*512*7*7)
        self.bottleneck = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)

        # skip connection and transition convolution
        self.skip4 = self._make_connection(skip_connection, 512, 512, 256, transition_up=True)
        self.skip3 = self._make_connection(skip_connection, 512, 256, 256, transition_up=True)
        self.skip2 = self._make_connection(skip_connection, 512, 128, 128, transition_up=True)
        self.skip1 = self._make_connection(skip_connection, 256, 64, 64, transition_up=True)
        self.skip0 = self._make_connection(skip_connection, 128, 64, 32, transition_up=True)

        self.conv_end_1 = nn.Conv2d(64, 384, kernel_size=3,
                                    stride=1, padding=1, bias=False)
        self.bn_end_1 = nn.BatchNorm2d(384)
        self.conv_end_3 = nn.Conv2d(2,32,kernel_size= 3,stride=1,padding=1,bias=False)
        self.conv_end_2 = nn.Conv2d(384, 384, kernel_size=1, bias=False)
        self.conv_end_4 = nn.Conv2d(32,2,kernel_size= 3, stride=1,padding=1,bias=False)
        self.bn_end_2 = nn.BatchNorm2d(384)

        self.LSoftmax = nn.LogSoftmax()


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


    def _make_connection(self, skip_connection, in_planes, skip_planes, out_planes_half, transition_up=False):
        layers = []
        layers.append(skip_connection(in_planes, skip_planes, out_planes_half, transition_up))

        return nn.Sequential(*layers)


    def forward(self, x, y):
        skip_connect = []

        x = self.conv1_ch(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        skip_connect.append(x)

        x = self.layer1(x)
        skip_connect.append(x)
        x = self.layer2(x)
        skip_connect.append(x)
        x = self.layer3(x)
        skip_connect.append(x)
        x = self.layer4(x)
        skip_connect.append(x)

        # bottleneck
        fea_map = self.bottleneck(x)

        # skip connection and transition convolution
        skip = skip_connect.pop()
        x = self.skip4((fea_map, skip))
        skip = skip_connect.pop()
        x = self.skip3((x, skip))
        skip = skip_connect.pop()
        x = self.skip2((x, skip))
        skip = skip_connect.pop()
        x = self.skip1((x, skip))
        skip = skip_connect.pop()
        x = self.skip0((x, skip))
        x = center_crop(x, 224, 224)

        x = self.conv_end_1(x)
        x = self.bn_end_1(x)
        x = self.relu(x)
        x = self.conv_end_2(x)
        x = self.bn_end_2(x)
        # x = self.LSoftmax()

        # add yourself
        y = self.conv_end_3(y)
        y = self.spnNet(x,y)
        x = self.spnNet(x,y)
        x = self.conv_end_4(x)
        x = self.LSoftmax(x)
        #print("x size", x.size())
        return x

	###############################spn###############
    def spnNet(self,out, y):
        Glr1 = out[:, 0:32]
        Glr2 = out[:, 32:64]
        Glr3 = out[:, 64:96]
        sum_abs = Glr1.abs() + Glr2.abs() + Glr3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        Glr1_norm = torch.div(Glr1, sum_abs)
        Glr2_norm = torch.div(Glr2, sum_abs)
        Glr3_norm = torch.div(Glr3, sum_abs)

        Glr1 = torch.add(-mask_need_norm, 1) * Glr1 + mask_need_norm * Glr1_norm
        Glr2 = torch.add(-mask_need_norm, 1) * Glr2 + mask_need_norm * Glr2_norm
        Glr3 = torch.add(-mask_need_norm, 1) * Glr3 + mask_need_norm * Glr3_norm

        ylr = y.cuda()
        Glr1 = Glr1.cuda()
        Glr2 = Glr2.cuda()
        Glr3 = Glr3.cuda()
        Propagator = GateRecurrent2dnoind(True, False)

        ylr = Propagator.forward(ylr, Glr1, Glr2, Glr3)

        Grl1 = out[:, 96:128];
        Grl2 = out[:, 128:160];
        Grl3 = out[:, 160:192];
        sum_abs = Grl1.abs() + Grl2.abs() + Grl3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        Grl1_norm = torch.div(Grl1, sum_abs)
        Grl2_norm = torch.div(Grl2, sum_abs)
        Grl3_norm = torch.div(Grl3, sum_abs)

        Grl1 = torch.add(-mask_need_norm, 1) * Grl1 + mask_need_norm * Grl1_norm
        Grl2 = torch.add(-mask_need_norm, 1) * Grl2 + mask_need_norm * Grl2_norm
        Grl3 = torch.add(-mask_need_norm, 1) * Grl3 + mask_need_norm * Grl3_norm

        yrl = y.cuda()
        Grl1 = Grl1.cuda()
        Grl2 = Grl2.cuda()
        Grl3 = Grl3.cuda()
        Propagator = GateRecurrent2dnoind(False,True)

        yrl = Propagator.forward(yrl, Grl1, Grl2, Grl3)

        Gdu1 = out[:, 192:224];
        Gdu2 = out[:, 224:256];
        Gdu3 = out[:, 256:288];
        sum_abs = Gdu1.abs() + Gdu2.abs() + Gdu3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        Gdu1_norm = torch.div(Gdu1, sum_abs)
        Gdu2_norm = torch.div(Gdu2, sum_abs)
        Gdu3_norm = torch.div(Gdu3, sum_abs)

        Gdu1 = torch.add(-mask_need_norm, 1) * Gdu1 + mask_need_norm * Gdu1_norm
        Gdu2 = torch.add(-mask_need_norm, 1) * Gdu2 + mask_need_norm * Gdu2_norm
        Gdu3 = torch.add(-mask_need_norm, 1) * Gdu3 + mask_need_norm * Gdu3_norm

        ydu = y.cuda()
        Gdu1 = Gdu1.cuda()
        Gdu2 = Gdu2.cuda()
        Gdu3 = Gdu3.cuda()
        Propagator = GateRecurrent2dnoind(False, False)

        ydu = Propagator.forward(ydu, Gdu1, Gdu2, Gdu3)

        Gud1 = out[:, 288:320];
        Gud2 = out[:, 320:352];
        Gud3 = out[:, 352:384];
        sum_abs = Gud1.abs() + Gud2.abs() + Gud3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        Gud1_norm = torch.div(Gud1, sum_abs)
        Gud2_norm = torch.div(Gud2, sum_abs)
        Gud3_norm = torch.div(Gud3, sum_abs)

        Gud1 = torch.add(-mask_need_norm, 1) * Gud1 + mask_need_norm * Gud1_norm
        Gud2 = torch.add(-mask_need_norm, 1) * Gud2 + mask_need_norm * Gud2_norm
        Gud3 = torch.add(-mask_need_norm, 1) * Gud3 + mask_need_norm * Gud3_norm

        yud = y.cuda()
        Gud1 = Gud1.cuda()
        Gud2 = Gud2.cuda()
        Gud3 = Gud3.cuda()
        Propagator = GateRecurrent2dnoind(True, True)

        yud = Propagator.forward(yud, Gud1, Gud2, Gud3)

        yout = torch.max(ylr,yrl)
        yout = torch.max(yout,yud)
        yout = torch.max(yout,ydu)
        #print("yout size", yout.size())
        #print("Gud1 size",Glr1.size())
        return yout


def resnet18(batchsize=1):

    model = ResNet(BasicBlock, [2, 2, 2, 2], SkipConnection, batchsize=batchsize)

    return model

