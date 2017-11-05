import torch as pt


class NormConv2d(pt.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, momentum=0.1, eps=1e-05, affine=True):
        super(NormConv2d, self).__init__()
        self.conv = pt.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = pt.nn.BatchNorm2d(
            out_channels, momentum=momentum, eps=eps, affine=affine)

    def forward(self, x):
        return self.norm(self.conv(x))


class NormLinear(pt.nn.Module):

    def __init__(self, in_features, out_features, bias=True, momentum=0.1, eps=1e-05, affine=True):
        super(NormLinear, self).__init__()
        self.fc = pt.nn.Linear(in_features, out_features, bias=bias)
        self.norm = pt.nn.BatchNorm1d(out_features, eps, momentum, affine)

    def forward(self, x):
        return self.norm(self.fc(x))


class ContinuousConv2d(pt.nn.Module):

    def __init__(self, channel_list, kernel_list, padding_list=None, stride_list=None,
                 conv=pt.nn.Conv2d, afunc=pt.nn.ReLU, **other):
        super(ContinuousConv2d, self).__init__()
        if padding_list is None:
            padding_list = [int(x // 2) for x in kernel_list]
        if stride_list is None:
            stride_list = [1 for _ in range(len(kernel_list))]
        conv_list = []
        for i in range(len(channel_list) - 1):
            conv_list.append(conv(
                channel_list[i], channel_list[i + 1], kernel_list[i],
                stride=stride_list[i], padding=padding_list[i], **other))
            conv_list.append(afunc())
        self.conv = pt.nn.Sequential(*conv_list[:-1])

    def forward(self, x):
        return self.conv(x)


class ContinuousLinear(pt.nn.Module):

    def __init__(self, feature_list, bias=True, linear=pt.nn.Linear, afunc=pt.nn.ReLU, **other):
        super(ContinuousLinear, self).__init__()
        linear_list = []
        for i in range(len(feature_list) - 1):
            linear_list.append(linear(
                feature_list[i], feature_list[i + 1], bias=bias, **other))
            linear_list.append(afunc())
        self.fc = pt.nn.Sequential(*linear_list[:-1])

    def forward(self, x):
        return self.fc(x)
