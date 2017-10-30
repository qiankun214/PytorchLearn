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
