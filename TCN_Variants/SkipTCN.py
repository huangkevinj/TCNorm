import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


############################################################################### How to use Skip TCN

class SeeInDark(nn.Module):
    def __init__(self, nf=8):
        super(SeeInDark, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.inn = InvBlock(nf,nf//2)
        self.dinn = DInvBlock(nf, nf // 2)

        self.conv1_1 = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(nf, nf, 2, stride=2)
        self.conv6_1 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(nf, nf, 2, stride=2)
        self.conv7_1 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(nf, nf, 2, stride=2)
        self.conv8_1 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)


        self.upv9 = nn.ConvTranspose2d(nf, nf, 2, stride=2)
        self.conv9_1 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(nf, 3, kernel_size=1, stride=1)



    def forward(self, x):
        conv1ori = self.conv1_1(x)
        conv1 = self.lrelu(self.conv1_2(self.lrelu(conv1ori)))
        xd,xsk,u,s = self.inn(conv1)

        pool1 = self.pool1(conv1)+xd

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = F.interpolate(self.upv6(conv5),size=(conv4.shape[2],conv4.shape[3]),mode='bilinear')
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = F.interpolate(self.upv7(conv6),size=(conv3.shape[2],conv3.shape[3]),mode='bilinear')
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = F.interpolate(self.upv8(conv7),size=(conv2.shape[2],conv2.shape[3]),mode='bilinear')
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))

        xr = self.dinn(conv8, xsk, u, s)

        up9 = F.interpolate(self.upv9(conv8),size=(conv1.shape[2],conv1.shape[3]),mode='bilinear')+xr
        up9 = torch.cat([up9, conv1], 1)


        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))


        conv10 = self.conv10_1(conv9)
        out = conv10

        return out



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt




############################################################################### Skip TCN



##################################################################################### When Used Downsampling



class InvBlock(nn.Module):
    def __init__(self, channel_num, out_channel, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        # self.split_len1 = channel_split_num  # 1
        # self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = identity()
        self.G = mean_operator()
        self.H = std_operator()

        if channel_num * 2 == out_channel:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Conv2d(2 * channel_num, out_channel, 1, 1, 0)

        # in_channels = 3
        # self.invconv = InvertibleConv1x1(channel_num, LU_decomposed=True)
        # self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x):
        # if not rev:
        #     # invert1x1conv
        #     # x, logdet = self.flow_permutation(x, logdet=0, rev=False)
        #
        #     # split to 1 channel and 2 channel.
        #     x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        #
        #     y1 = x1 + self.F(x2)  # 1 channel
        #     self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        #     y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        #     out = torch.cat((y1, y2), 1)
        # else:
        # split.
        orix = x
        _, _, H, W = x.shape
        xs1 = x[:, :, 0::2, 0::2]
        xs2 = x[:, :, 1::2, 0::2]
        xs3 = x[:, :, 0::2, 1::2]
        xs4 = x[:, :, 1::2, 1::2]
        x1, x2 = torch.cat([xs1, xs4], 1), torch.cat([xs2, xs3], 1)
        self.s = self.H(x1)
        y2 = (x2 - self.G(x1)).div(self.s)
        y1 = x1 - self.F(y2)

        y1 = self.conv(y1)
        # y1unnorm = torch.pixel_shuffle(y1, 2)
        # y2norm = torch.pixel_shuffle(y2, 2)
        # x = torch.cat((y1, y2), 1)
        # out = x
        # inv permutation
        # out, logdet = self.flow_permutation(x, logdet=0, rev=True)

        return y1, y2, self.G(x1), self.H(x1)


##################################################################################### When Used Upsampling


class DInvBlock(nn.Module):
    def __init__(self, channel_num, skip_channel, clamp=0.8):
        super(DInvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        # self.split_len1 = channel_split_num  # 1
        # self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        if channel_num == skip_channel:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Conv2d(channel_num, skip_channel, 1, 1, 0)

        self.F = identity()
        self.G = mean_operator()
        self.H = std_operator()

    def forward(self, xu, xsk, u, s):
        # if not rev:
        #     # invert1x1conv
        #     # x, logdet = self.flow_permutation(x, logdet=0, rev=False)
        #
        #     # split to 1 channel and 2 channel.
        #     x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        #
        #     y1 = x1 + self.F(x2)  # 1 channel
        #     self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        #     y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        #     out = torch.cat((y1, y2), 1)
        # else:
        # split.
        xu = self.conv(xu)
        xu = xu + self.F(xsk)
        xsk = xsk.mul(s) + u

        xout = torch.cat((xu, xsk), 1)
        out = torch.pixel_shuffle(xout, 2)

        return out


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self,x):
        return x


class mean_operator(nn.Module):
    def __init__(self):
        super(mean_operator, self).__init__()

    def forward(self,x):
        out = mean_channels(x)
        return out


class std_operator(nn.Module):
    def __init__(self):
        super(std_operator, self).__init__()

    def forward(self, x):
        out = stdv_channels(x)
        return out



def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
