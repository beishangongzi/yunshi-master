import torch
import torch.nn as nn
import torch.nn.functional as F

from util import get_map_val


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
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
    def __init__(self, in_channels, out_channels, att_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.att_block = attention_block(out_channels, out_channels, att_channels)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, out, out_up):
        out_up1 = self.up(out_up)
        out_up1 = F.interpolate(out_up1, size=(out.size(2), out.size(3)),
                            mode="bilinear", align_corners=True)
        out_up1 = self.att_block(g=out_up1, x=out)
        out = torch.cat([out_up1, out], dim=1)
        out = self.up_conv(out)
        return out


class attention_block(nn.Module):
    def __init__(self, g_c, x_c, out_c):
        super(attention_block, self).__init__()
        self.g_conv = nn.Sequential(
            nn.Conv2d(g_c, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c)
        )
        self.x_conv = nn.Sequential(
            nn.Conv2d(x_c, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c)
        )
        self.p_conv = nn.Sequential(
            nn.Conv2d(out_c, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.g_conv(g)
        x1 = self.x_conv(x)
        p = self.relu(g1+x1)
        p = self.p_conv(p)
        return x * p


class AttentionUnet(nn.Module):
    def __init__(self, ch_in=5, ch_out=7):
        super(AttentionUnet, self).__init__()
        self.down1 = encoder(ch_in, 64)
        self.down2 = encoder(64, 128)
        self.down3 = encoder(128, 256)
        self.down4 = encoder(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = decoder(1024, 512, 256)
        self.up2 = decoder(512, 256, 128)
        self.up3 = decoder(256, 128, 64)
        self.up4 = decoder(128, 64, 32)
        self.final_conv = nn.Conv2d(64, ch_out, kernel_size=1)

    def forward(self, x):
        x1, x1_down = self.down1(x)
        x2, x2_down = self.down2(x1_down)
        x3, x3_down = self.down3(x2_down)
        x4, x4_down = self.down4(x3_down)

        middle = self.middle_conv(x4_down)

        up1 = self.up1(x4, middle)
        up2 = self.up2(x3, up1)
        up3 = self.up3(x2, up2)
        up4 = self.up4(x1, up3)

        out = self.final_conv(up4)
        return out


def create_model(config):
    ch_in = get_map_val(config, 'ch_in', 3)
    ch_out = get_map_val(config, 'ch_out', 7)
    return AttentionUnet(ch_in=ch_in, ch_out=ch_out)


if __name__ == '__main__':
    x = torch.rand((1,3,256,256))
    model = AttentionUnet(3, 7)
    y = model(x)
    print(y.size())