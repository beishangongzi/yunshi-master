import torch
import torch.nn as nn
import torch.nn.functional as F

from util import get_map_val


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        out = self.down_conv(x)
        out_down = self.pool(out)
        return out, out_down


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        # 使用反卷积实现上采样
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, out, out_up):
        out_up = self.up(out_up)
        out_up = F.interpolate(out_up, size=(out.size(2), out.size(3)),
                            mode="bilinear", align_corners=True)

        out = torch.cat([out, out_up], dim=1)
        out = self.up_conv(out)
        return out


class UNetPlusPlus(nn.Module):
    def __init__(self, ch_in=5, ch_out=7):
        super(UNetPlusPlus, self).__init__()

        self.down0_0 = encoder(ch_in, 64)
        self.down1_0 = encoder(64, 128)
        self.down2_0 = encoder(128, 256)
        self.down3_0 = encoder(256, 512)
        self.middle4_0 = encoder(512, 1024)

        self.middle0_1 = decoder(128, 64)
        self.middle0_2 = decoder(128, 64)
        self.middle0_3 = decoder(128, 64)

        self.middle1_1 = decoder(256, 128)
        self.middle1_2 = decoder(256, 128)

        self.middle2_1 = decoder(512, 256)

        self.up3_1 = decoder(1024, 512)
        self.up2_2 = decoder(512, 256)
        self.up1_3 = decoder(256, 128)
        self.up0_4 = decoder(128, 64)

        self.final_conv = nn.Conv2d(64, ch_out, kernel_size=1)

    def forward(self, x):
        out0_0, out_down0_0 = self.down0_0(x)
        out1_0, out_down1_0 = self.down1_0(out_down0_0)
        out2_0, out_down2_0 = self.down2_0(out_down1_0)
        out3_0, out_down3_0 = self.down3_0(out_down2_0)

        out4_0, _ = self.middle4_0(out_down3_0)

        out0_1 = self.middle0_1(out0_0, out1_0)
        out1_1 = self.middle1_1(out1_0, out2_0)
        out2_1 = self.middle2_1(out2_0, out3_0)

        out0_2 = self.middle0_2(out0_1, out1_1)
        out1_2 = self.middle1_2(out1_1, out2_1)

        out0_3 = self.middle0_3(out0_2, out1_2)

        out3_1 = self.up3_1(out3_0, out4_0)
        out2_2 = self.up2_2(out2_1, out3_1)
        out1_3 = self.up1_3(out1_2, out2_2)
        out0_4 = self.up0_4(out0_3, out1_3)

        out = self.final_conv(out0_4)
        return out


def create_model(config):
    ch_in = get_map_val(config, 'ch_in', 3)
    ch_out = get_map_val(config, 'ch_out', 7)
    return UNetPlusPlus(ch_in=ch_in, ch_out=ch_out)


if __name__ == '__main__':
    x = torch.rand((1,3,256,256))
    model = UNetPlusPlus(3, 7)
    y = model(x)
    print(y.size())