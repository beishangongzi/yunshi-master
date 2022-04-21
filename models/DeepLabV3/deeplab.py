import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DeepLabV3.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.DeepLabV3.aspp import build_aspp
from models.DeepLabV3.decoder import build_decoder
from models.DeepLabV3.backbone import build_backbone

from util import get_map_val


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, ch_in=3, ch_out=21,
                 sync_bn=True, freeze_bn=False, pretrained=True):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_ch=ch_in, pretrained=pretrained)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(ch_out, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


def create_model(config):
    ch_in = get_map_val(config, 'ch_in', 3)
    ch_out = get_map_val(config, 'ch_out', 7)
    return DeepLab(ch_in=ch_in, ch_out=ch_out)


if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16, ch_in=5, pretrained=False)
    model.eval()
    input = torch.rand(1, 5, 256, 256)
    output = model(input)
    print(output.size())


