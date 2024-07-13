import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from PIL import Image

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()
        ratio = 2
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # # [N, 1, H*W]
        # context = torch.matmul(avg_x, theta_x)     ####### B,1,C * B,C,HW
        # # [N, 1, H*W]
        # context = self.softmax_left(context)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        context = torch.matmul(avg_x, theta_x)


        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out

class PSA_s(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_s, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # # [N, C, H, W]
        # out = self.spatial_pool(x)
        # # [N, C, H, W]
        # out = self.channel_pool(out)

        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel

        return out

#
# class disparity_attention(nn.Module):
#     def __init__(self, inplanes, kernel_size=1, stride=1):
#         super(disparity_attention, self).__init__()
#         planes = inplanes
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.inplanes = inplanes
#         self.inter_planes = planes // 2
#         self.planes = planes
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = (kernel_size - 1) // 2
#         ratio = 2
#
#         self.conv_q = nn.Conv3d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
#         self.conv_k = nn.Conv3d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
#                                       bias=False)
#         self.conv_v = nn.Conv3d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
#                                       bias=False)
#
#         self.conv_up = nn.Sequential(
#             nn.Conv3d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
#             nn.BatchNorm3d(self.inter_planes // ratio),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(self.inter_planes // ratio, self.planes, kernel_size=1)
#         )
#         self.softmax = nn.Softmax(dim=-1)
#         self.sigmoid = nn.Sigmoid()
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         kaiming_init(self.conv_q, mode='fan_in')
#         kaiming_init(self.conv_k, mode='fan_in')
#         kaiming_init(self.conv_v, mode='fan_in')
#
#         self.conv_q.inited = True
#         self.conv_k.inited = True
#         self.conv_v.inited = True
#
#     def disparity_aggregate(self, x):
#         # [B, C, D, H, W]
#         query = self.conv_q(x)
#
#         batch, channel, disparity, height, width = query.size()
#
#         # [B, D, C,H,W]
#         query = query.permute(0,2,1,3,4)
#         # [B, D, CHW]
#         query = query.reshape(batch, disparity, channel*height*width)
#
#         # [B, C, D, H ,W]
#         key = self.conv_k(x)
#         # [B, C, H, W, D]
#         key = key.permute(0,1,3,4,2)
#         # [B, CHW, D]
#         key = key.reshape(batch, channel*height*width, disparity)
#
#         # [B, D, D]
#         energy = torch.matmul(query, key)
#         range_e = energy.max()-energy.min()
#         energy = (energy-energy.min()) / range_e
#         # energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
#
#         # [B, D, D]
#         attention = self.softmax(energy)
#
#         # [B, C, D, H ,W]
#         value = self.conv_v(x)
#         # [B, D, C, H ,W]
#         value = value.permute(0,2,1,3,4)
#         # [B, D, CHW]
#         value = value.reshape(batch, disparity, channel*height*width)
#
#         # [B, D, CHW]
#         out = torch.matmul(attention, value)
#         # [B, D, C, H ,W]
#         out = out.reshape(batch, disparity, channel, height, width)
#         # [B, C, D, H ,W]
#         out = out.permute(0,2,1,3,4)
#
#         out = self.conv_up(out)
#         out = self.gamma*out + x
#
#         return out
#
#     def forward(self, x):
#
#         out = self.disparity_aggregate(x)
#
#         return out


class disparity_attention2(nn.Module):
    def __init__(self, inplanes, kernel_size=1, stride=1):
        super(disparity_attention2, self).__init__()
        planes = inplanes
        self.gamma = nn.Parameter(torch.zeros(1))
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 2

        self.conv_q = nn.Conv3d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_k = nn.Conv3d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        self.conv_v = nn.Conv3d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)

        self.conv_up = nn.Sequential(
            nn.Conv3d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.BatchNorm3d(self.inter_planes // ratio),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)


        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q, mode='fan_in')
        kaiming_init(self.conv_k, mode='fan_in')
        kaiming_init(self.conv_v, mode='fan_in')

        self.conv_q.inited = True
        self.conv_k.inited = True
        self.conv_v.inited = True

    def channel_aggregate(self, x):
        # [B, C, D, H, W]
        query = self.conv_q(x)

        batch, channel, disparity, height, width = query.size()

        # [B, D, C,H,W]
        query = query.permute(0,2,1,3,4)
        # [B, D, CHW]
        query = query.reshape(batch, disparity, channel*height*width)

        # [B, C, D, H ,W]
        key = self.conv_k(x)
        # [B, C, H, W, D]
        key = key.permute(0,1,3,4,2)
        # [B, CHW, D]
        key = key.reshape(batch, channel*height*width, disparity)

        # [B, D, D]
        energy = torch.matmul(query, key)
        range_e = energy.max()-energy.min()
        energy = (energy-energy.min()) / range_e
        # energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        # [B, D, D]
        attention = self.softmax(energy)

        # [B, C, D, H ,W]
        value = self.conv_v(x)
        # [B, D, C, H ,W]
        value = value.permute(0,2,1,3,4)
        # [B, D, CHW]
        value = value.reshape(batch, disparity, channel*height*width)

        # [B, D, CHW]
        out = torch.matmul(attention, value)
        # [B, D, C, H ,W]
        out = out.reshape(batch, disparity, channel, height, width)
        # [B, C, D, H ,W]
        out = out.permute(0,2,1,3,4)

        out = self.conv_up(out)
        out = self.gamma*out + x

        return out


    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):

        out = self.disparity_aggregate(x)

        return out


class my_unary(nn.Module):
    def __init__(self, inplanes, kernel_size=1, stride=1):
        super(my_unary, self).__init__()

        planes = inplanes
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)

        self.conv_upl = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )

        # self.conv_upr = nn.Sequential(
        #     nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
        #     nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        # )

        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()


        self.gamma_left = nn.Parameter(torch.zeros(1))
        self.gamma_right = nn.Parameter(torch.zeros(1))
        self.leftout = nn.Sequential(nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes), nn.ReLU())
        self.rightout = nn.Sequential(nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inplanes), nn.ReLU())


        # self.allout = nn.Sequential(nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
        #                             nn.BatchNorm2d(inplanes), nn.ReLU())

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)
        # [N, C, 1, 1]
        context = self.conv_upl(context)


        out = x + context.expand_as(x) * self.gamma_right

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        out = x + context.expand_as(x) * self.gamma_left

        return out

    def forward(self, x):

        # [N, C, H, W]
        context_channel = self.spatial_pool(x)


        ########## [N, C, H, W]
        context_spatial = self.channel_pool(x)


        ##########[N, C, H, W]
        out =  self.leftout(context_spatial) + self.rightout(context_channel)
        # out = self.leftout(context_spatial)


        return out


if __name__ == '__main__':

    q = torch.randn(1, 16, 32, 64).cuda()  # B C H W

    model = my_unary2(inplanes=16).cuda()
    for i in range(10000):
        x0 = model(q)
        print(x0.shape)