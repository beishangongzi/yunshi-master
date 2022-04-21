import torch
import torch.nn as nn

from util import get_map_val


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(ch_out),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        return self.conv(x)


class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(ch_in, ch_out)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
            )
        self.double_conv = conv_block(ch_in, ch_out)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2],dim=1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, ch_in=3, ch_out=21):
        super(UNet, self).__init__()
        self.num_class = ch_out

        self.down_block1 = conv_block(ch_in, 64)
        self.down_block2 = down_conv(64, 128)
        self.down_block3 = down_conv(128, 256)
        self.down_block4 = down_conv(256, 512)
        self.down_block5 = down_conv(512, 1024)

        self.up_block5 = up_conv(1024, 512)
        self.up_block4 = up_conv(512, 256)
        self.up_block3 = up_conv(256, 128)
        self.up_block2 = up_conv(128, 64)

        self.conv_1_1 = nn.Conv2d(64, ch_out,1)

    def forward(self, x):
        # extracting path
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        # expanding path
        d1 = self.up_block5(x5, x4)
        d2 = self.up_block4(d1, x3)
        d3 = self.up_block3(d2, x2)
        d4 = self.up_block2(d3, x1)
        d5 = self.conv_1_1(d4)

        return d5


def create_model(config):
    ch_in = get_map_val(config, 'ch_in', 3)
    ch_out = get_map_val(config, 'ch_out', 7)
    return UNet(ch_in=ch_in, ch_out=ch_out)
