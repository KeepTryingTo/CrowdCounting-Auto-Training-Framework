# -*- coding: utf-8 -*-
"""P2R 模型与数据增强工具

- UpSample_P2P / SimpleDecoder / convblock：vendor 自 P2RLoss-main/models/utils.py
  （结构未改动）
- GaussianBlur / NormalSample：vendor 自 P2RLoss-main/datasets/utils.py，
  NormalSample 增补 augment / resize_factor 两个参数以对接框架的
  「数据增强」开关与「增强强度」（关闭增强时保留随机裁剪——组 batch
  必需——但取消随机缩放 / 翻转 / 强增强）
"""
import random

import torch
import torch.nn.functional as F
from PIL import ImageFilter
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ------------------------- 模型结构件 -------------------------

def convblock(inc, ouc, kernel_size, bn=True):
    padding = kernel_size // 2
    module = torch.nn.Sequential(
        torch.nn.Conv2d(inc, ouc, kernel_size=kernel_size, stride=1,
                        padding=padding, bias=not bn),
        torch.nn.BatchNorm2d(ouc) if bn else torch.nn.Identity(),
        torch.nn.ReLU(inplace=True)
    )
    if not bn:
        torch.nn.init.constant_(module[0].bias, 0.)
    return module


def conv_3x3(inc, ouc, bn=True):
    return convblock(inc, ouc, kernel_size=3, bn=bn)


class UpSample_P2P(torch.nn.Module):
    """FPN 式融合：各层 1×1 对齐到首层尺寸后逐元素相加 + 3×3 融合"""

    def __init__(self, incs, ouc, bn=True, relu=True):
        super().__init__()
        self.align_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(inc, ouc, kernel_size=1, stride=1, padding=0,
                            bias=not bn) for inc in incs
        ])
        if not bn:
            for layer in self.align_layers:
                torch.nn.init.constant_(layer.bias, 0.)

        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(ouc, ouc, kernel_size=3, stride=1, padding=1,
                            bias=not bn),
            torch.nn.BatchNorm2d(ouc) if bn else torch.nn.Identity(),
            torch.nn.ReLU(inplace=True) if relu else torch.nn.Identity()
        )

        self.fuse_channel = ouc

    def forward(self, xs):
        x0 = self.align_layers[0](xs[0])
        out_shape = x0.shape[-2:]
        for x, layer in zip(xs[1:], self.align_layers[1:]):
            x = F.interpolate(layer(x), out_shape, mode='bilinear',
                              align_corners=False)
            x0 = x0 + x
        x = self.fuse(x0)
        return x


class SimpleDecoder(torch.nn.Sequential):
    def __init__(self, in_channel=128, fea_channel=64, up_scale=1,
                 out_channel=1):
        super().__init__(
            conv_3x3(in_channel, fea_channel, bn=False),
            conv_3x3(fea_channel, fea_channel, bn=False),
            torch.nn.Conv2d(fea_channel, out_channel * (up_scale ** 2),
                            kernel_size=3, stride=1, padding=1),
            torch.nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale
        torch.nn.init.constant_(self[-2].bias, 0.)


# ------------------------- 数据增强 -------------------------

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class NormalSample(object):
    """P2R 训练样本预处理器（vendor 自 P2RLoss-main，适配框架增强开关）

    - im2tensor：弱增强（ToTensor + ImageNet 归一化）——教师模型输入
    - strong_aug：强增强（色彩抖动 / 灰度 / 高斯模糊 + 归一化）——学生模型
      半监督输入（FixMatch 风格弱-强视图对）
    - process_lable：训练时随机缩放 + 裁剪 crop_size + 翻转（图像与坐标
      同步），32 倍数 padding；坐标输出 (y, x, 最近邻距离²) 三列
      （与源实现一致，损失仅用前两列）；评估时不裁剪不翻转，仅 padding
    - process_unlabel：无标注弱/强视图对的裁剪 + 翻转 + padding
    """

    def __init__(self, mean=None, std=None, crop_size=(256, 256),
                 resize_factor=0.3, train=False, augment=True):
        mean = mean or IMAGENET_MEAN
        std = std or IMAGENET_STD
        self.half_h, self.half_w = crop_size
        self.train = train
        self.augment = augment
        self.scale_factor = max(0.0, min(1.0, resize_factor))

        self.im2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        if augment:
            self.strong_aug = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.25),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.8),
                self.im2tensor
            ])
        else:
            # 增强关闭：强视图退化为弱视图（半监督管线仍可运行）
            self.strong_aug = self.im2tensor

    def process_lable(self, image, dotseq):
        if self.train:
            images, dotseqs = self.crop_and_resize(image, dotseq)
        else:
            images, dotseqs = image.unsqueeze(0), [dotseq]

        h, w = images.shape[-2:]
        # 高宽非 32 倍数则填充（编码器 /8 + PixelShuffle ×2 → /4 须整除）
        if h % 32 != 0 or w % 32 != 0:
            ph = (32 - h % 32) % 32
            pw = (32 - w % 32) % 32
            images = F.pad(images, (0, pw, 0, ph))
            h, w = images.shape[-2:]
        for i in range(images.size(0)):
            if self.train and self.augment and random.randint(0, 1):
                images[i] = torch.flip(images[i], dims=(-1,))
                dotseqs[i][:, 0] = w - dotseqs[i][:, 0] - 1

        for i, seq in enumerate(dotseqs):
            u = self.nearest(seq)
            # 坐标列交换 (x,y) → (y,x)（与损失中 (row,col) 网格对齐），拼接最近邻距离²
            dotseqs[i] = torch.cat((seq[:, [1, 0]], u), dim=1)

        return images, dotseqs

    def process_unlabel(self, image):
        if self.train:
            images = self.crop_and_resize(image)
        else:
            raise NotImplementedError("process_unlabel 仅用于训练")

        h, w = images.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            ph = (32 - h % 32) % 32
            pw = (32 - w % 32) % 32
            images = F.pad(images, (0, pw, 0, ph))

        for i in range(images.size(0)):
            if self.train and self.augment and random.randint(0, 1):
                images[i] = torch.flip(images[i], dims=(-1,))
        return images

    def crop_and_resize(self, image, dotseq=None, num_patches=1):
        """随机缩放（scale ∈ [1-f, 1+f]）+ 随机裁剪 crop_size + 插值回裁剪尺寸

        image: [3,H,W] 或 [6,H,W]（无标注弱强视图拼接）；dotseq: [N,2] (x,y)
        """
        imh, imw = image.shape[-2:]

        scale = (random.random() * (self.scale_factor * 2)
                 + (1 - self.scale_factor)) if self.augment else 1.0
        crop_h = int(self.half_h / scale + 0.5)
        crop_w = int(self.half_w / scale + 0.5)

        if crop_h > imh or crop_w > imw:
            padw = crop_w - imw
            padh = crop_h - imh
            image = F.pad(image, (0, padw, 0, padh), mode='constant', value=0)
            imh, imw = image.shape[-2:]

        crop_imgs, crop_dots = [], []
        rh, rw = self.half_h / crop_h, self.half_w / crop_w
        for _ in range(num_patches):
            start_h = random.randint(0, imh - crop_h)
            start_w = random.randint(0, imw - crop_w)
            end_h = start_h + crop_h
            end_w = start_w + crop_w
            crop_img = image[:, start_h:end_h, start_w:end_w]
            crop_imgs.append(crop_img)

            if dotseq is not None:
                idx = ((dotseq[:, 0] >= start_w) & (dotseq[:, 0] <= end_w) &
                       (dotseq[:, 1] >= start_h) & (dotseq[:, 1] <= end_h))
                selected_dot = dotseq[idx]
                selected_dot = selected_dot.clone()
                selected_dot[:, 0] = (selected_dot[:, 0] - start_w) * rw
                selected_dot[:, 1] = (selected_dot[:, 1] - start_h) * rh
                crop_dots.append(selected_dot)

        crop_imgs = torch.stack(crop_imgs, dim=0)
        crop_imgs = F.interpolate(crop_imgs, (self.half_h, self.half_w),
                                  mode='bilinear', align_corners=False)

        if dotseq is not None:
            return crop_imgs, crop_dots
        return crop_imgs

    @staticmethod
    def nearest(seq):
        """每点到最近邻点的欧氏距离平方（P2RLoss 源实现，损失未使用此列，
        保留以维持 vendor 数据格式一致）"""
        seqlen = seq.size(0)
        if seqlen <= 1:
            return torch.zeros(seqlen, 1) + 32
        xx = (seq ** 2).sum(dim=-1, keepdim=True)
        xy = seq @ seq.T
        yy = xx.T
        L2 = (xx - 2 * xy + yy).relu()
        m = torch.kthvalue(L2, 2, dim=1).values
        return m.view(-1, 1)
