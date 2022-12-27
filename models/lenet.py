"""
https://wenku.baidu.com/view/1dbf69ddf9b069dc5022aaea998fcc22bcd143a9.html
"""
import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """

    def __init__(self, num_classes=10, dropout=0.0, kernel_nums=None, activation=None, **kwargs):
        super(LeNet5, self).__init__()
        if activation is None:
            activation = lambda: nn.ReLU(inplace=True)

        if kernel_nums is None:
            kernel_nums = [6, 16, 120]
        self.kernel_nums = kernel_nums
        self.c1 = nn.Conv2d(1, kernel_nums[0], kernel_size=(5, 5), padding=2)
        self.s2 = nn.Sequential(activation(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.c3 = nn.Conv2d(kernel_nums[0], kernel_nums[1], kernel_size=(5, 5))
        self.s4 = nn.Sequential(activation(), nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.c5 = nn.Conv2d(kernel_nums[1], kernel_nums[2], kernel_size=(5, 5))
        self.act_c5 = activation()
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(kernel_nums[2], 84)
        self.act_f6 = activation()
        self.drop = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.output = nn.Linear(84, num_classes)

    def forward(self, img):
        x = self.c1(img)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.c5(x)
        x = self.act_c5(x)
        x = self.f6(self.flatten(x))
        x = self.act_f6(x)
        x = self.drop(x)
        output = self.output(x)
        return output


def lenet5(**kwargs):
    return LeNet5(**kwargs)


def lenet5f20_c3(**kwargs):
    """
    fit 20x20 image by setting padding of c3 to 2,
    without change the numel of parameters
    """
    net = LeNet5(**kwargs)
    net.c3 = nn.Conv2d(net.kernel_nums[0], net.kernel_nums[1], kernel_size=(5, 5), padding=2)
    return net


def lenet5f20_c3k3(**kwargs):
    """
    fit 20x20 image by setting kernel size of c3 to (3,3)
    without change the numel of parameters
    """
    net = LeNet5(**kwargs)
    net.c3 = nn.Conv2d(net.kernel_nums[0], net.kernel_nums[1], kernel_size=(3, 3), padding=1)
    return net


def lenet5f20_c5(**kwargs):
    """
    fit 20x20 image by setting padding of c5 to 2,
    without change the numel of parameters
    """
    net = LeNet5(**kwargs)
    net.c5 = nn.Conv2d(net.kernel_nums[1], net.kernel_nums[2], kernel_size=(5, 5), padding=1)
    return net


def lenet5f20_c5k3(**kwargs):
    """
    fit 20x20 image by setting kernel size of c5 to (3,3)
    """
    net = LeNet5(**kwargs)
    net.c5 = nn.Conv2d(net.kernel_nums[1], net.kernel_nums[2], kernel_size=(3, 3))
    return net


def lenet5hk(**kwargs):
    """
    half of kernels [3, 16, 60] (1011/2022)
    """
    return LeNet5(kernel_nums=[3, 16, 60], **kwargs)


def lenet5hk2(**kwargs):
    """
    half of kernels [6, 8, 120] (1014/2022)
    """
    return LeNet5(kernel_nums=[6, 8, 120], **kwargs)


def lenet5hk3(**kwargs):
    """
    half of kernels [6, 8, 119] (1006/2022)
    """
    return LeNet5(kernel_nums=[6, 8, 119], **kwargs)


def lenet5half(**kwargs):
    return LeNet5(kernel_nums=[3, 8, 60], **kwargs)


def lenet5_lkrelu(**kwargs):
    return LeNet5(activation=lambda: nn.LeakyReLU(inplace=True), **kwargs)


def lenet5_tanh(**kwargs):
    return LeNet5(activation=nn.Tanh, **kwargs)


def try_fix20():
    def print_try_fix20(lenet5f20_func):
        net = lenet5f20_func()
        print(lenet5f20_func.__name__,
              net(torch.zeros(32, 1, 20, 20)).shape,
              sum(p.numel() for p in net.parameters()))

    print_try_fix20(lenet5f20_c3)
    print_try_fix20(lenet5f20_c3k3)
    print_try_fix20(lenet5f20_c5)
    print_try_fix20(lenet5f20_c5k3)


def try_lenet():
    def print_try(lenet5_func):
        net = lenet5_func()
        print(lenet5_func.__name__,
              net(torch.zeros(32, 1, 28, 28)).shape,
              sum(p.numel() for p in net.parameters()))

    print_try(lenet5)
    print_try(lenet5_lkrelu)
    print_try(lenet5_tanh)
    print_try(lenet5hk)
    print_try(lenet5hk2)
    print_try(lenet5hk3)


if __name__ == '__main__':
    """
    """
    try_fix20()
    try_lenet()
    # print(sum(p.numel() for p in net.parameters()))
    # print(6 + 6 * 16 + 16 * 120)
    # print(3 + 3 * 16 + 16 * 60)
    # print(6 + 6 * 8 + 8 * 120)
    # print(6 + 6 * 8 + 8 * 119)
