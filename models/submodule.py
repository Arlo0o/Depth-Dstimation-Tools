#-*-coding:gb2312-*-
from __future__ import print_function
import os
import sys

import torch
import cv2
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from torch.nn.utils import weight_norm

def trans_color(gray):
    gray = gray.reshape(gray.shape[0], gray.shape[1], 1).astype('uint8')
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    return out



def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume



class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SmoothL1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, disp0, disp1, disp2, target):
        loss0 = F.smooth_l1_loss(disp0, target)
        loss1 = F.smooth_l1_loss(disp1, target)
        loss2 = F.smooth_l1_loss(disp2, target)
        return loss0, loss1, loss2


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True),
                                   )

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left = F.pad(
            torch.index_select(left, 3, Variable(torch.LongTensor([i for i in range(shift, width)])).cuda()),
            (shift, 0, 0, 0))
        shifted_right = F.pad(
            torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width - shift)])).cuda()),
            (shift, 0, 0, 0))
        out = torch.cat((shifted_left, shifted_right), 1).view(batch, filters * 2, 1, height, width)
        return out


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda(),
                             requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


class sepConv3dBlock(nn.Module):
    '''
    Separable 3d convolution block as 2 separable convolutions and a projection
    layer
    '''

    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        super(sepConv3dBlock, self).__init__()
        if in_planes == out_planes and stride == (1, 1, 1):
            self.downsample = None
        else:
            self.downsample = projfeat3d(in_planes, out_planes, stride)
        self.conv1 = sepConv3d(in_planes, out_planes, 3, stride, 1)
        self.conv2 = sepConv3d(out_planes, out_planes, 3, (1, 1, 1), 1)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out), inplace=True)
        return out


class projfeat3d(nn.Module):
    '''
    Turn 3d projection into 2d projection
    '''

    def __init__(self, in_planes, out_planes, stride):
        super(projfeat3d, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, out_planes, (1, 1), padding=(0, 0), stride=stride[:2], bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        b, c, d, h, w = x.size()
        x = self.conv1(x.view(b, c, d, h * w))
        x = self.bn(x)
        x = x.view(b, -1, d // self.stride[0], h, w)
        return x
def sepConv3d(in_planes, out_planes, kernel_size, stride, pad, bias=False):
    if bias:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=bias))
    else:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=bias),
            nn.BatchNorm3d(out_planes))
class decoderBlock(nn.Module):
    def __init__(self, nconvs, inchannelF, channelF, stride=(1, 1, 1), up=False, nstride=1, pool=False):
        super(decoderBlock, self).__init__()
        self.pool = pool
        stride = [stride] * nstride + [(1, 1, 1)] * (nconvs - nstride)
        self.convs = [sepConv3dBlock(inchannelF, channelF, stride=stride[0])]
        for i in range(1, nconvs):
            self.convs.append(sepConv3dBlock(channelF, channelF, stride=stride[i]))
        self.convs = nn.Sequential(*self.convs)

        self.classify = nn.Sequential(sepConv3d(channelF, channelF, 3, (1, 1, 1), 1),
                                      nn.ReLU(inplace=True),
                                      sepConv3d(channelF, 1, 3, (1, 1, 1), 1, bias=True))

        self.up = False
        if up:
            self.up = True
            self.up = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
                                    sepConv3d(channelF, channelF // 2, 3, (1, 1, 1), 1, bias=False),
                                    nn.ReLU(inplace=True))

        if pool:
            self.pool_convs = torch.nn.ModuleList([sepConv3d(channelF, channelF, 1, (1, 1, 1), 0),
                                                   sepConv3d(channelF, channelF, 1, (1, 1, 1), 0),
                                                   sepConv3d(channelF, channelF, 1, (1, 1, 1), 0),
                                                   sepConv3d(channelF, channelF, 1, (1, 1, 1), 0)])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.xavier_uniform(m.weight)
                # torch.nn.init.constant(m.weight,0.001)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm3d):
            #    m.weight.data.fill_(1)
            #    m.bias.data.zero_()
            #    m.running_mean.data.fill_(0)
            #    m.running_var.data.fill_(1)

    def forward(self, fvl):
        # left
        fvl = self.convs(fvl)
        # pooling
        if self.pool:
            fvl_out = fvl
            _, _, d, h, w = fvl.shape
            for i, pool_size in enumerate(np.linspace(1, min(d, h, w) // 2, 4, dtype=int)):
                kernel_size = (int(d / pool_size), int(h / pool_size), int(w / pool_size))
                out = F.avg_pool3d(fvl, kernel_size, stride=kernel_size)
                out = self.pool_convs[i](out)
                out = F.upsample(out, size=(d, h, w), mode='trilinear')
                fvl_out = fvl_out + 0.25 * out
            fvl = F.relu(fvl_out / 2., inplace=True)

        # #TODO cost aggregation
        # costl = self.classify(fvl)
        # if self.up:
        #     fvl = self.up(fvl)
        if self.training:
            # classification
            costl = self.classify(fvl)
            if self.up:
                fvl = self.up(fvl)
        else:
            # classification
            if self.up:
                fvl = self.up(fvl)
                costl = fvl
            else:
                costl = self.classify(fvl)

        return fvl, costl.squeeze(1)



class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        # self.unet = UNet_3Plus_DeepSup(3, 128)
        # self.mytransformer = CvT()

        # self.atten1 = DisentangledNonLocal2d(in_channels=64, temperature=0.05, reduction=2)
        # self.atten2 = DisentangledNonLocal2d(in_channels=128, temperature=0.05, reduction=2)
        # self.atten3 = DisentangledNonLocal2d(in_channels=128, temperature=0.05, reduction=2)

        self.se1 = nn.Sequential(
                                convbn(64, 64, 1, 1, 0, 1),
                                nn.ReLU(inplace=True),
                                SELayer(64, 16),
                                 convbn(64, 64, 1, 1, 0, 1),
                                 )
        self.se2 = nn.Sequential(
                                convbn(128, 128, 1, 1, 0, 1),
                                nn.ReLU(inplace=True),
                                SELayer(128, 16),
                                 convbn(128, 128, 1, 1, 0, 1),
                                 )
        self.se3 = nn.Sequential(
                                convbn(128, 128, 1, 1, 0, 1),
                                nn.ReLU(inplace=True),
                                SELayer(128, 16),
                                 convbn(128, 128, 1, 1, 0, 1),
                                 )

        self.atrous_block1 = nn.Sequential(convbn(320, 128, 1, 1, 0, 1),
                                           nn.ReLU(inplace=True),
                                           )
        self.atrous_block6 = nn.Sequential(convbn(320, 128, 3, 1, 2, dilation=2),
                                           nn.ReLU(inplace=True),
                                           )
        self.atrous_block12 = nn.Sequential(convbn(320, 128, 3, 1, 4, dilation=4),
                                            nn.ReLU(inplace=True),
                                            )
        self.atrous_block18 = nn.Sequential(convbn(320, 128, 3, 1, 6, dilation=6),
                                            nn.ReLU(inplace=True),
                                            )
        self.conv_1x1_output = nn.Sequential(convbn(128 * 4, 320, 1, 1, 0, 1),
                                             nn.ReLU(inplace=True),
                                             )



        self.inplanes = 32
        self.relu = nn.ReLU(inplace=True)
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),        
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),                         
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),                    
                                       )


        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)  # 输出通道数，每个basicblock有两层
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                     convbn(in_planes=320, out_planes=128, kernel_size=3, stride=1, pad = 1, dilation=1),
                                     nn.ReLU(inplace=True),
                                     convbn(in_planes=128, out_planes=128, kernel_size=1, stride=1, pad=0, dilation=1),
                                    #  nn.ReLU(inplace=True),
                                     )

        self.branch2 = nn.Sequential(nn.AvgPool2d((2, 2), stride=(2, 2)),
                                     convbn(in_planes=320, out_planes=128, kernel_size=3, stride=1, pad = 1, dilation=1),
                                     nn.ReLU(inplace=True),
                                     convbn(in_planes=128, out_planes=128, kernel_size=1, stride=1, pad=0, dilation=1),
                                    #  nn.ReLU(inplace=True),
                                     )

        self.branch3 = nn.Sequential(
                                     convbn(in_planes=320, out_planes=128, kernel_size=3, stride=1, pad = 1, dilation=1),
                                     nn.ReLU(inplace=True),
                                     # convbn(in_planes=128, out_planes=128, kernel_size=1, stride=1, pad=0, dilation=1),
                                    #  nn.ReLU(inplace=True),
                                    )


        self.lastconv1 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 12, kernel_size=1, padding=0, stride = 1, bias=False))
        self.lastconv2 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 12, kernel_size=1, padding=0, stride = 1, bias=False))
        self.lastconv3 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 12, kernel_size=1, padding=0, stride = 1, bias=False))



    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        l1_1 = self.firstconv(x)  ## 1/2
        l1_1 = self.layer1(l1_1)

        l2_1 = self.layer2(l1_1) ## 1/4
        l3_1 = self.layer3(l2_1)
        l4_1 = self.layer4(l3_1)
        # l4_1 = self.atten3(l4_1)

        # l2_2, l3_2, l4_2 = self.atten1(l2_1), self.atten2(l3_1), self.atten3(l4_1)
        # l2_2, l3_2, l4_2 = F.interpolate(l2_2, [l2_1.size()[2], l2_1.size()[3]], mode='bilinear'), \
        #                    F.interpolate(l3_2, [l2_1.size()[2], l2_1.size()[3]], mode='bilinear'), \
        #                    F.interpolate(l4_2, [l2_1.size()[2], l2_1.size()[3]], mode='bilinear')
        # l2 = self.se1(torch.cat((l2_1, l2_2), dim=1))
        # l3 = self.se2(torch.cat((l3_1, l3_2), dim=1))
        # l4 = self.se3(torch.cat((l4_1, l4_2), dim=1))
        # l2, l3, l4 = self.se1(l2_1),self.se2(l3_1),self.se3(l4_1)
        gwc_feature = torch.cat((l2_1, l3_1, l4_1), dim=1)


        # gwc_feature2 = self.conv_1x1_output( torch.cat((self.atrous_block1(gwc_feature), self.atrous_block6(gwc_feature),
        #                                              self.atrous_block12(gwc_feature), self.atrous_block18(gwc_feature)),  dim=1) )


        output_branch1 = self.branch1(gwc_feature)  # 这里实验spp 1/16
        output_branch2 = self.branch2(gwc_feature)  # 1/8
        output_branch3 = self.branch3(gwc_feature)  # 1/4


        # gwc_feature = self.unet(x)
        # output_branch1 = gwc_feature[4]
        # output_branch2 = gwc_feature[3]
        # output_branch3 = gwc_feature[2]
        # low = gwc_feature[5]


        output_branch1_ = self.lastconv1(output_branch1)
        output_branch2_ = self.lastconv2(output_branch2)
        output_branch3_ = self.lastconv3(output_branch3)

        return [output_branch1, output_branch1_, output_branch2, output_branch2_,  output_branch3,  output_branch3_,l1_1]  # 16 8 4


class MY_NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(MY_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.AvgPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.AvgPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.AvgPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        # self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            # self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,kernel_size=1, stride=1, padding=0)
            self.W = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, q, k, v):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = q.size(0)

        g_x = self.g(v).view(batch_size, self.inter_channels, -1)  #### 认为是v，进行下采样
        g_x = g_x.permute(0, 2, 1)  ###


        theta_x = self.theta(k).view(batch_size, self.inter_channels, -1)  ##### 认为是k,不进行下采样
        theta_x = theta_x.permute(0, 2, 1)  ###
        phi_x = self.phi(q).view(batch_size, self.inter_channels, -1)  ##### 认为是q，进行下采样
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)


        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *q.size()[2:])
        W_y = self.W(y)
        # z = W_y + v
        z = W_y + v
        return z
class MYNONLocalBlock2D(MY_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(MYNONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
class MYNONLocalBlock3D(MY_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(MYNONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

if __name__ == '__main__':
    a = torch.randn(1, 32, 128, 128).cuda()  # B C D H W  1/8
    b = torch.randn(1, 3, 48, 64, 128).cuda()
    c = torch.randn(1, 600, 128*256).cuda()


    for i in range(1000):
        net = MYNONLocalBlock2D(32).cuda()
        out = net(a,a,a)
        print(out.size())
