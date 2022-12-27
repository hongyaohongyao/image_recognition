import torch
import torch.nn as nn
from torchvision.models import alexnet


# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         self.convs = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return torch.concat([self.m(x), x], dim=1)

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


def mcnn(num_classes=10, init_weights=True, **kwargs):
    widths = [16, 64, 128]
    model = nn.Sequential(
        nn.Conv2d(1, widths[0], kernel_size=(5, 5), padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(widths[0], widths[1], kernel_size=(5, 5), padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(widths[1], widths[2], kernel_size=(5, 5), padding=2),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(widths[2], num_classes)
    )
    if init_weights:
        _init_weights(model)
    return model


def catmcnn(num_classes=10, init_weights=True, **kwargs):
    widths = [16, 64, 128]
    model = nn.Sequential(
        nn.Conv2d(1, widths[0], kernel_size=(5, 5), padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        ConcatBlock(nn.Sequential(
            nn.Conv2d(widths[0], widths[1], kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True)
        )),
        nn.MaxPool2d(kernel_size=3, stride=2),
        ConcatBlock(nn.Sequential(
            nn.Conv2d(widths[1] + widths[0], widths[2], kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True)
        )),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(widths[2] + widths[1] + widths[0], num_classes)
    )
    if init_weights:
        _init_weights(model)
    return model


def main():
    def print_try(model_func):
        net = model_func()
        print(model_func.__name__,
              net(torch.zeros(32, 1, 28, 28)).shape,
              sum(p.numel() for p in net.parameters()))

    print_try(mcnn)
    print_try(catmcnn)


if __name__ == '__main__':
    main()
