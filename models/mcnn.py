import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResBlock(nn.Module):

    def __init__(self, m):
        super(ResBlock, self).__init__()
        self.m = m

    def forward(self, x):
        return self.m(x) + x


class ConcatBlock(nn.Module):

    def __init__(self, m):
        super(ConcatBlock, self).__init__()
        self.m = m

    def forward(self, x):
        return torch.concat([self.m(x), x], dim=1)


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode="fan_out",
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(
                m,
            (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def downsample(in_channels, out_channels):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


def conv_bn_relu(in_channels, out_channels):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


def conv_bn(channels):
    return [
        nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(channels),
    ]


def mcnn(num_classes=10,
         channels=1,
         blocks_num=None,
         init_weights=True,
         **kwargs):
    widths = [64, 128, 256]

    def blocks(channels, block_num=2):
        res = []
        for _ in range(block_num):
            res.extend([
                ResBlock(nn.Sequential(*conv_bn(channels))),
                nn.ReLU(inplace=True)
            ])
        return res

    model = nn.Sequential(*[
        *conv_bn_relu(channels, widths[0]), *blocks(widths[0], blocks_num[0]),
        *downsample(widths[0], widths[1]), *blocks(widths[1], blocks_num[1]),
        *downsample(widths[1], widths[2]), *blocks(widths[2], blocks_num[2]),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(widths[2], num_classes)
    ])
    if init_weights:
        _init_weights(model)
    return model


def resmcnn(num_classes=10,
            channels=1,
            blocks_num=None,
            init_weights=True,
            **kwargs):
    widths = [64, 128, 256]

    def blocks(channels, block_num=2):
        res = []
        for _ in range(block_num):
            res.extend([
                ResBlock(nn.Sequential(*conv_bn(channels))),
                nn.ReLU(inplace=True)
            ])
        return res

    model = nn.Sequential(*[
        *conv_bn_relu(channels, widths[0]), *blocks(widths[0], blocks_num[0]),
        *downsample(widths[0], widths[1]), *blocks(widths[1], blocks_num[1]),
        *downsample(widths[1], widths[2]), *blocks(widths[2], blocks_num[2]),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(widths[2], num_classes)
    ])
    if init_weights:
        _init_weights(model)
    return model


def mcnn10(**kwargs):
    return mcnn(blocks_num=[2, 2, 2], **kwargs)


def mcnn16(**kwargs):
    return mcnn(blocks_num=[4, 4, 4], **kwargs)


def mcnn28(**kwargs):
    return mcnn(blocks_num=[8, 8, 8], **kwargs)


def resmcnn10(**kwargs):
    return resmcnn(blocks_num=[2, 2, 2], **kwargs)


def resmcnn16(**kwargs):
    return resmcnn(blocks_num=[4, 4, 4], **kwargs)


def resmcnn28(**kwargs):
    return resmcnn(blocks_num=[8, 8, 8], **kwargs)


def main():

    def print_try(model_func):
        net = model_func()
        print(model_func.__name__,
              net(torch.zeros(32, 1, 28, 28)).shape,
              sum(p.numel() for p in net.parameters()))

    print_try(mcnn10)
    print_try(resmcnn10)
    print_try(mcnn16)
    print_try(resmcnn16)
    print_try(mcnn28)
    print_try(resmcnn28)


if __name__ == '__main__':
    main()
