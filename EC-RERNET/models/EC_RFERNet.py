import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.data.size(0), -1)


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU6, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        #c_weight = self.gate_fn(x_se)
        return x
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        self.inp1 = inp // 3
        self.inp2 = inp - self.inp1
        #init_channels = math.ceil(oup / ratio)
        #new_channels = init_channels*(ratio-1)

        if self.inp1 <16:
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(self.inp1, self.inp1, dw_size, 1, dw_size // 2, groups=1, bias=False),
                nn.BatchNorm2d(self.inp1),
            )
        else:
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(self.inp1, self.inp1, dw_size, 1, dw_size//2, groups=1, bias=False),
                nn.BatchNorm2d(self.inp1),
                nn.ReLU6(inplace=True) if relu else nn.Sequential(),
            )

        if self.oup < 16:
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, self.oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(self.oup),
            )
        else:
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, self.oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(self.oup),
                nn.ReLU6(inplace=True) if relu else nn.Sequential(),
            )

        self.se = SqueezeExcite(self.oup)



    def forward(self, x):
        # x1 = self.primary_conv(x)
        # x2 = self.cheap_operation(x1)
        # out = torch.cat([x1,x2], dim=1)

        x1, x2 = torch.split(x,[self.inp1,self.inp2], dim=1)
        x1 = self.cheap_operation(x1)
        x = torch.cat([x1,x2], dim=1)

        x = self.primary_conv(x)
        x = self.se(x)

        return x
class GhostModule1(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule1, self).__init__()
        self.oup = oup
        self.inp1 = inp // 3
        self.inp2 = inp - self.inp1
        #init_channels = math.ceil(oup / ratio)
        #new_channels = init_channels*(ratio-1)

        if self.inp1 <16:
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(self.inp1, self.inp1, dw_size, 1, dw_size // 2, groups=1, bias=False),
                nn.BatchNorm2d(self.inp1),
            )
        else:
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(self.inp1, self.inp1, dw_size, 1, dw_size//2, groups=1, bias=False),
                nn.BatchNorm2d(self.inp1),
                nn.ReLU6(inplace=True) if relu else nn.Sequential(),
            )

        if self.oup < 16:
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, self.oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(self.oup),
            )
        else:
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, self.oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(self.oup),
                nn.ReLU6(inplace=True) if relu else nn.Sequential(),
            )





    def forward(self, x):
        # x1 = self.primary_conv(x)
        # x2 = self.cheap_operation(x1)
        # out = torch.cat([x1,x2], dim=1)

        x1, x2 = torch.split(x,[self.inp1,self.inp2], dim=1)
        x1 = self.cheap_operation(x1)
        x = torch.cat([x1,x2], dim=1)

        x = self.primary_conv(x)


        return x

class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, out_channels, stride=stride))

    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels

        groups = in_channels
        kernel = 3
        # print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')

        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                            stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))

    def forward(self, x):
        return super().forward(x)
class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,
                                          stride=stride, padding=kernel // 2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))

    def forward(self, x):
        return super().forward(x)


class ConvLayer1(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        out_ch = out_channels
        groups = 1
        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', GhostModule(in_channels, out_ch, kernel_size=kernel, ratio=2, dw_size=3, stride=1, relu=True))
        #self.add_module('relu', nn.ReLU6(True))

    def forward(self, x):
        return super().forward(x)

class ConvLayer2(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        out_ch = out_channels
        groups = 1
        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', GhostModule1(in_channels, out_ch, kernel_size=kernel, ratio=2, dw_size=3, stride=1, relu=True))
        #self.add_module('relu', nn.ReLU6(True))

    def forward(self, x):
        return super().forward(x)

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            #print('outch',outch)
            if (i % 2 == 0) :
                layers_.append(ConvLayer1(inch, outch))
            else:
                layers_.append(ConvLayer1(inch, outch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
                #print('self.out_channels',self.out_channels)
        # print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                # print('x',x.size())
                # print('tin',len(tin))
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out
class HarDBlock1(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            #print('outch',outch)
            if (i % 2 == 0) :
                layers_.append(ConvLayer2(inch, outch))
            else:
                layers_.append(ConvLayer2(inch, outch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
                #print('self.out_channels',self.out_channels)
        # print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                # print('x',x.size())
                # print('tin',len(tin))
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out

class EC_RFERNet(nn.Module):
    def __init__(self, depth_wise=False, arch=39, pretrained=False, weight_path=''):
        super().__init__()
        second_kernel = 3
        max_pool = True
        drop_rate = 0.1
        first_ch = [24, 48]
        ch_list = [96, 160, 240]
        #ch_list = [96, 160, 240]
        grmul = 1.6
        gr = [16, 20, 64]
        #gr = [16, 20, 64]
        n_layers = [4, 8,4]
        downSamp = [1, 1,0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        self.base.append(
            ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                      stride=1, bias=False))

        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))
        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))
        ch = first_ch[1]
        for i in range(blks):
            if i == 0 :
                blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            else:
                blk = HarDBlock1(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i == blks - 1 and arch == 85:
                self.base.append(nn.Dropout(0.1))

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))

                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))
        ch = ch_list[blks - 1]
        self.base.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(ch, 7)))

        # print(self.base)
        for m in self.named_parameters():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        for layer in self.base:
            x = layer(x)

        return x



