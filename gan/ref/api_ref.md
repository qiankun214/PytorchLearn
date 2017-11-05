# GAN中可能需要的API资料

## Generator部分

### 反卷积函数

> *class*`torch.nn.``ConvTranspose2d`(*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, *output_padding=0*, *groups=1*, *bias=True*, *dilation=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#ConvTranspose2d)

Applies a 2D transposed convolution operator over an input image composed of several input planes.

This module can be seen as the gradient of Conv2d with respect to its input. It is also known as a fractionally-strided convolution or a deconvolution (although it is not an actual deconvolution operation).

- `stride` controls the stride for the cross-correlation.
- If `padding` is non-zero, then the input is implicitly zero-padded on both sides for `padding`number of 			points.
- If `output_padding` is non-zero, then the output is implicitly zero-padded on one side for `output_padding` number of points.
- `dilation` controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation` does.
- `groups` controls the connections between inputs and outputs. in_channels and out_channelsmust both be divisible by groups.At groups=1, all inputs are convolved to all outputs.At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels, and producing half the output channels, and both subsequently concatenated. At groups=`in_channels`, each input channel is convolved with its own set of filters (of size out_channels // in_channels).

The parameters `kernel_size`, `stride`, `padding`, `output_padding` can either be

- a single `int` – in which case the same value is used for the height and width dimensions
- a `tuple` of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension

- **in_channels** ([*int*](https://docs.python.org/2/library/functions.html#int)) – Number of channels in the input image
- **out_channels** ([*int*](https://docs.python.org/2/library/functions.html#int)) – Number of channels produced by the convolution
- **kernel_size** ([*int*](https://docs.python.org/2/library/functions.html#int)* or *[*tuple*](https://docs.python.org/2/library/functions.html#tuple)) – Size of the convolving kernel
- **stride** ([*int*](https://docs.python.org/2/library/functions.html#int)* or *[*tuple*](https://docs.python.org/2/library/functions.html#tuple)*, **optional*) – Stride of the convolution. Default: 1
- **padding** ([*int*](https://docs.python.org/2/library/functions.html#int)* or *[*tuple*](https://docs.python.org/2/library/functions.html#tuple)*, **optional*) – Zero-padding added to both sides of the input. Default: 0
- **output_padding** ([*int*](https://docs.python.org/2/library/functions.html#int)* or *[*tuple*](https://docs.python.org/2/library/functions.html#tuple)*, **optional*) – Zero-padding added to one side of the output. Default: 0
- **groups** ([*int*](https://docs.python.org/2/library/functions.html#int)*, **optional*) – Number of blocked connections from input channels to output channels. Default: 1
- **bias** ([*bool*](https://docs.python.org/2/library/functions.html#bool)*, **optional*) – If True, adds a learnable bias to the output. Default: True
- **dilation** ([*int*](https://docs.python.org/2/library/functions.html#int)* or *[*tuple*](https://docs.python.org/2/library/functions.html#tuple)*, **optional*) – Spacing between kernel elements. Default: 1


- Shape:
  - Input: (N,Cin,Hin,Win)(N,Cin,Hin,Win)
  - Output: (N,Cout,Hout,Wout)(N,Cout,Hout,Wout) where 
    - Hout=(Hin−1)∗stride[0]−2∗padding[0]+kernel_size[0]+output_padding[0]
    - Wout=(Win−1)∗stride[1]−2∗padding[1]+kernel_size[1]+output_padding[1]

| Variables: | **weight** ([*Tensor*](http://pytorch.org/docs/master/tensors.html#torch.Tensor)) – the learnable weights of the module of shape (in_channels, out_channels, kernel_size[0], kernel_size[1]) |
| ---------- | ---------------------------------------- |
|            | **bias** ([*Tensor*](http://pytorch.org/docs/master/tensors.html#torch.Tensor)) – the learnable bias of the module of shape (out_channels) |

Examples:

```python
>>> # With square kernels and equal stride
>>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
>>> output = m(input)
>>> # exact output size can be also specified as an argument
>>> input = autograd.Variable(torch.randn(1, 16, 12, 12))
>>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
>>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
>>> h = downsample(input)
>>> h.size()
torch.Size([1, 16, 6, 6])
>>> output = upsample(h, output_size=input.size())
>>> output.size()
torch.Size([1, 16, 12, 12])
```