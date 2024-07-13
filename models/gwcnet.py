from __future__ import print_function
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import os
import sys
sys.path.append('.')
from models.submodule import *
from models.my_att import *




####################------------------------------best gwc-----------------------------#############################


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=True, concat_feature_channel=12):
        super(feature_extraction, self).__init__()

     

        self.concat_feature = concat_feature
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1, bias=False))

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

        l1 = self.firstconv(x)  ## 1/2
        l1 = self.layer1(l1)

        l2_1 = self.layer2(l1) ## 1/4
        l3_1 = self.layer3(l2_1)
        l4_1 = self.layer4(l3_1)

        gwc_feature = torch.cat((l2_1, l3_1, l4_1), dim=1)
        # l2_1, l3_1, l4_1 = self.se1(l2_1), self.se2(l3_1), self.se3(l4_1)


        if not self.concat_feature:
            return {"gwc_feature": gwc_feature, "low": l1}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature,"low": l1}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))


        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))


        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))


        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)   ### 1/8

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)  ####  1/16

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, expansion_ratio: int):
        super(ResBlock, self).__init__()
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats+16 , n_feats * expansion_ratio, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x: torch.Tensor, disp: torch.Tensor):
        return x + self.module(torch.cat([disp, x], dim=1))


class GwcNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=True):
        super(GwcNet, self).__init__()

        #################--------unary atten修改------------###################
        self.myunary = my_unary(inplanes=32)
     

        self.adjust1 = nn.Sequential(convbn(32, 16, 3, 1, 1, 1),
                                                  )


        self.basic0_0 = nn.Sequential(convbn(1, 16, 3, 1, 1, 1),
                                      convbn(16, 16, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True), )
        self.basic0_1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.basic0_2 = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn(32, 1, 1, 1, 0, 1),
                                      )

        self.basic1_0 = nn.Sequential(convbn(1, 16, 3, 1, 1, 1),
                                      convbn(16, 16, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True), )
        self.basic1_1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.basic1_2 = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn(32, 1, 1, 1, 0, 1),
                                      )

        self.basic2_0 = nn.Sequential(convbn(1, 16, 3, 1, 1, 1),
                                      convbn(16, 16, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True), )
        self.basic2_1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.basic2_2 = nn.Sequential(
                                      convbn(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn(32, 1, 1, 1, 0, 1),
                                      )

        self.basic3_0 = nn.Sequential(convbn(1, 16, 3, 1, 1, 1),
                                      convbn(16, 16, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      )
        self.basic3_1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.basic3_2 = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn(32, 1, 1, 1, 0, 1),
                                       )

        self.deconv = nn.Sequential(convbn(8, 8, 3, 1, 1, 1),
                                    nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=5, padding=2,
                                                       output_padding=3, stride=4, bias=False),
                                    )


        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        self.inplanes = planes * block.expansion
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, left, right):

        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        cost0 = self.classif0(cost0)
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2)
        cost3 = self.classif3(out3)

        cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost0 = torch.squeeze(cost0, 1)
        pred0 = F.softmax(cost0, dim=1)
        pred0 = disparity_regression(pred0, self.maxdisp)

        cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost1 = torch.squeeze(cost1, 1)
        pred1 = F.softmax(cost1, dim=1)
        pred1 = disparity_regression(pred1, self.maxdisp)

        cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost2 = torch.squeeze(cost2, 1)
        pred2 = F.softmax(cost2, dim=1)
        pred2 = disparity_regression(pred2, self.maxdisp)

        cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparity_regression(pred3, self.maxdisp)

        ################################################-------refinement---------##################################
        low = features_left["low"]
        low = self.myunary(low)



        low = F.upsample(low, [left.size()[2], left.size()[3]], mode='bilinear')

        low = self.adjust1(low)



        pred0_ = self.basic0_0(pred0.unsqueeze(1))
        low0 = torch.cat((low, pred0_), dim=1)
        low0 = self.basic0_1(low0)
        low0 = self.basic0_2(low0)
        pred0_2 = pred0 + low0.squeeze(1)

        pred1_ = self.basic1_0(pred1.unsqueeze(1))
        low1 = torch.cat((low, pred1_), dim=1)
        low1 = self.basic1_1(low1)
        low1 = self.basic1_2(low1)
        pred1_2 = pred1 + low1.squeeze(1)

        pred2_ = self.basic2_0(pred2.unsqueeze(1))
        low2 = torch.cat((low, pred2_), dim=1)
        low2 = self.basic2_1(low2)
        low2 = self.basic2_2(low2)
        pred2_2 = pred2 + low2.squeeze(1)

        pred3_ = self.basic3_0(pred3.unsqueeze(1))
        low3 = torch.cat((low, pred3_), dim=1)
        low3 = self.basic3_1(low3)
        low3 = self.basic3_2(low3)
        pred3_2 = pred3 + low3.squeeze(1)



        if self.training:
            return pred0_2, pred1_2, pred2_2, pred3_2

        else:
            return pred3_2