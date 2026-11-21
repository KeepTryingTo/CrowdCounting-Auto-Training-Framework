# Backbone module, adapted from P2PNet.
from torch import nn

from . import vgg_


class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str,
                 return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])   # 128ch @ 1/2
                self.body2 = nn.Sequential(*features[13:23]) # 256ch @ 1/4
                self.body3 = nn.Sequential(*features[23:33]) # 512ch @ 1/8
                self.body4 = nn.Sequential(*features[33:43]) # 512ch @ 1/16
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, x):
        out = []
        if self.return_interm_layers:
            xs = x
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)
        else:
            xs = self.body(x)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    def __init__(self, name: str, return_interm_layers: bool,
                 pretrained=True, weights_path=None):
        if name == 'vgg16_bn':
            backbone = vgg_.vgg16_bn(pretrained=pretrained, weights_path=weights_path)
        else:
            raise ValueError('unsupported backbone: {}'.format(name))
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


def build_backbone(args):
    backbone = Backbone_VGG(
        name=args.backbone,
        return_interm_layers=True,
        pretrained=not args.no_pretrained,
        weights_path=getattr(args, 'vgg_weights', None))
    return backbone
