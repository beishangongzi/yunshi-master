import sys, os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from itertools import chain
from util import get_map_val

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class ResNet(nn.Module):
    '''
    -> ResNet BackBone
    '''
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16: s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8: s3, s4, d3, d4 = (1, 1, 2, 4)
        
        if output_stride == 8: 
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        self.final_conv = nn.Conv2d(in_dim * 2, in_dim, 3, padding=1)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        x = torch.cat(out, 1)
        return self.final_conv(x)


class Decorder(nn.Module):
    '''
    -> Decoder
    '''
    def __init__(self, ch_in, ch_out):
        super(Decorder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out * 2, ch_out, 3, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, xr):
        x1 = self.conv1(x)

        # 尺寸对齐
        if xr is not None:
            x1_s = x1.size()
            xr_s = xr.size()
            if x1_s[2] != xr_s[2] or x1_s[3] != xr_s[3]:
                x1 = F.interpolate(x1, size=xr_s[2:], mode='bilinear', align_corners=True)

            # 特征图融合
            x1 = torch.cat((x1, xr), dim=1)
            x1 = self.conv2(x1)
        return x1


'''
-> Deeplab V3 +
'''
configs = {
    'resnet101': {'aspp_ch_in': 2048, 'channels': [2048, 1024, 512, 256, 64]},
    'resnet34': {'aspp_ch_in': 512, 'channels': []},
}


class DeepLab(nn.Module):
    def __init__(self, ch_in=3, ch_out=2, pretrained=False,
                output_stride=16, backbone='resnet101', **_):
        config = configs[backbone]
        channels = config['channels']

        super(DeepLab, self).__init__()
        
        self.backbone = ResNet(in_channels=ch_in, output_stride=output_stride, pretrained=pretrained, backbone=backbone)

        bins = (1, 2, 3, 6)
        self.ppm = PPM(channels[0], int(2048/len(bins)), bins)

        # self.decoder = Decoder(low_level_channels, ch_out)
        self.decoder1 = Decorder(channels[0], channels[1])
        self.decoder2 = Decorder(channels[1], channels[2])
        self.decoder3 = Decorder(channels[2], channels[3])
        self.decoder4 = Decorder(channels[3], channels[4])

        self.final_conv = nn.Conv2d(channels[4], ch_out, 1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        x1, x2, x3, x4, x5 = self.backbone(x)
        x5_ = self.ppm(x5)
        # xs = [x1, x2, x3, x4, x5]
        # for x in xs:
        #     print('xs', x.size())
        u1 = self.decoder1(x5_, x4)
        u2 = self.decoder2(u1, x3)
        u3 = self.decoder3(u2, x2)
        u4 = self.decoder4(u3, x1)

        x = F.interpolate(u4, size=(H, W), mode='bilinear', align_corners=True)
        return self.final_conv(x)

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


def create_model(config):
    ch_in = get_map_val(config, 'ch_in', 3)
    ch_out = get_map_val(config, 'ch_out', 7)
    backbone = get_map_val(config, 'backbone', 'resnet101')
    return DeepLab(ch_in=ch_in, ch_out=ch_out, backbone=backbone)


if __name__ == '__main__':
    model = DeepLab(3)
    # model.freeze_bn()
    # model.eval()
    model.train()
    image = torch.autograd.Variable(torch.randn(2, 3, 512, 512), volatile=True)
    # print(type(model.resnet_features))
    result = model(image)
    print(result.size())
    # print(result[1].size())