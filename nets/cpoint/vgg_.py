# VGG definition, adapted from P2PNet/torchvision.
# Pretrained weights are resolved in this order:
#   1. explicit --vgg_weights path
#   2. local fallback D:\...\Codes\weights\pretrained\vgg16_bn.pth
#   3. torch hub cache / auto download (vgg16_bn-6c64b313.pth)
import os

import torch
import torch.nn as nn

__all__ = ['vgg16_bn']

model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

# Local fallback: Codes/weights/pretrained/vgg16_bn.pth (2 levels up)
_LOCAL_FALLBACK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'weights', 'pretrained', 'vgg16_bn.pth')


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


def _load_vgg16_bn_weights(model, weights_path=None):
    """Resolve and load ImageNet-pretrained vgg16_bn weights."""
    candidates = []
    if weights_path:
        candidates.append(weights_path)
    if os.path.exists(_LOCAL_FALLBACK):
        candidates.append(os.path.abspath(_LOCAL_FALLBACK))
    for path in candidates:
        state_dict = torch.load(path, map_location='cpu')
        # torchvision classifiers keys are compatible with this VGG class
        model.load_state_dict(state_dict)
        print('=> loaded vgg16_bn pretrained weights from {}'.format(path))
        return model
    # fall back to torch hub cache / download
    try:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls['vgg16_bn'], progress=False, map_location='cpu')
        model.load_state_dict(state_dict)
        print('=> loaded vgg16_bn pretrained weights from torch hub cache/download')
        return model
    except Exception as e:  # noqa
        print('=> WARNING: could not load vgg16_bn pretrained weights ({}), '
              'training from scratch'.format(e))
        return model


def vgg16_bn(pretrained=True, weights_path=None):
    model = VGG(make_layers(cfgs['D'], batch_norm=True),
                init_weights=not pretrained)
    if pretrained:
        model = _load_vgg16_bn_weights(model, weights_path)
    return model
