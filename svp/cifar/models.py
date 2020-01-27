import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs


class StochasticBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, survival_rate=1):
        super().__init__()
        self.survival_rate = survival_rate
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.increasing = inplanes != (planes * self.expansion)
        if self.increasing:
            assert ((1. * planes * self.expansion) / inplanes) == 2
        if stride != 1:
            self.shortcut = nn.Sequential(nn.AvgPool2d(stride))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        if self.increasing:
            shortcut = torch.cat([shortcut] + [shortcut.mul(0)], 1)

        if not self.training or torch.rand(1)[0] <= self.survival_rate:
            H = self.conv1(inputs)
            H = self.bn1(H)
            H = F.relu(H)

            H = self.conv2(H)
            H = self.bn2(H)

            if self.training:
                H /= self.survival_rate
            H += shortcut
        else:
            H = shortcut
        outputs = F.relu(H)

        return outputs


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)

        self.increasing = stride != 1 or inplanes != (planes * self.expansion)
        if self.increasing:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.bn1(inputs)
        H = F.relu(H)
        if self.increasing:
            inputs = H
        H = self.conv1(H)

        H = self.bn2(H)
        H = F.relu(H)
        H = self.conv2(H)

        H += self.shortcut(inputs)
        return H


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)
        H = F.relu(H)

        H = self.conv3(H)
        H = self.bn3(H)

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=32,
                 base_width=4):
        super().__init__()

        width = math.floor(planes * (base_width / 64.0))

        self.conv1 = nn.Conv2d(inplanes, width * cardinality, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * cardinality)

        self.conv2 = nn.Conv2d(width * cardinality, width * cardinality, 3,
                               groups=cardinality, padding=1, stride=stride,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(width * cardinality)

        self.conv3 = nn.Conv2d(width * cardinality, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)
        H = F.relu(H)

        H = self.conv3(H)
        H = self.bn3(H)

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, stride=stride,
                               bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)

        self.increasing = stride != 1 or inplanes != (planes * self.expansion)
        if self.increasing:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.bn1(inputs)
        H = F.relu(H)
        if self.increasing:
            inputs = H
        H = self.conv1(H)

        H = self.bn2(H)
        H = F.relu(H)
        H = self.conv2(H)

        H = self.bn3(H)
        H = F.relu(H)
        H = self.conv3(H)

        H += self.shortcut(inputs)
        return H


class ResNet(nn.Module):

    def __init__(self, Block, layers, filters, num_classes=10, inplanes=None):
        self.inplanes = inplanes or filters[0]
        super().__init__()

        self.pre_act = 'Pre' in Block.__name__

        self.conv1 = nn.Conv2d(3, self.inplanes, 3, padding=1, bias=False)
        if not self.pre_act:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.num_sections = len(layers)
        for section_index, (size, planes) in enumerate(zip(layers, filters)):
            section = []
            for layer_index in range(size):
                if section_index != 0 and layer_index == 0:
                    stride = 2
                else:
                    stride = 1
                section.append(Block(self.inplanes, planes, stride=stride))
                self.inplanes = planes * Block.expansion
            section = nn.Sequential(*section)
            setattr(self, f'section_{section_index}', section)

        if self.pre_act:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.fc = nn.Linear(filters[-1] * Block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')  # noqa: E501
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        H = self.conv1(inputs)

        if not self.pre_act:
            H = self.bn1(H)
            H = F.relu(H)

        for section_index in range(self.num_sections):
            H = getattr(self, f'section_{section_index}')(H)

        if self.pre_act:
            H = self.bn1(H)
            H = F.relu(H)

        H = F.avg_pool2d(H, H.size()[2:])
        H = H.view(H.size(0), -1)
        outputs = self.fc(H)

        return outputs


class StochasticResNet(ResNet):

    def __init__(self, Block, layers, filters, num_classes=10, inplanes=None,
                 min_survival_rate=1.0, decay='linear'):
        super().__init__(Block, layers, filters,
                         num_classes=num_classes,
                         inplanes=inplanes)
        L = sum(layers)
        curr = 1
        for section_index in range(self.num_sections):
            section = getattr(self, f'section_{section_index}')
            for name, module in section.named_children():
                if decay == 'linear':
                    survival_rate = 1 - ((curr / L) * (1 - min_survival_rate))
                elif decay == 'uniform':
                    survival_rate = min_survival_rate
                else:
                    raise NotImplementedError(
                        f"{decay} decay has not been implemented.")
                module.survival_rate = survival_rate
                curr += 1
        assert (curr - 1) == L


# From "Deep Residual Learning for Image Recognition"
def ResNet20(num_classes=10):
    return ResNet(BasicBlock, layers=[3] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet32(num_classes=10):
    return ResNet(BasicBlock, layers=[5] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet44(num_classes=10):
    return ResNet(BasicBlock, layers=[7] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet56(num_classes=10):
    return ResNet(BasicBlock, layers=[9] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet110(num_classes=10):
    return ResNet(BasicBlock, layers=[18] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet1202(num_classes=10):
    return ResNet(BasicBlock, layers=[200] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


# From "Identity Mappings in Deep Residual Networks"
def PreActResNet110(num_classes=10):
    return ResNet(PreActBlock, layers=[18] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet164(num_classes=10):
    return ResNet(PreActBottleneck, layers=[18] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet1001(num_classes=10):
    return ResNet(PreActBottleneck, layers=[111] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


# Based on but not in "Identity Mappings in Deep Residual Networks"
def PreActResNet8(num_classes=10):
    return ResNet(PreActBlock, layers=[1] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet14(num_classes=10):
    return ResNet(PreActBlock, layers=[2] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet20(num_classes=10):
    return ResNet(PreActBlock, layers=[3] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet56(num_classes=10):
    return ResNet(PreActBlock, layers=[9] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet164Basic(num_classes=10):
    return ResNet(PreActBlock, layers=[27] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


# From "Deep Networks with Stochastic Depth"
def StochasticResNet110(num_classes=10):
    return StochasticResNet(StochasticBlock, layers=[18] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear', num_classes=num_classes)


def StochasticResNet1202(num_classes=10):
    return StochasticResNet(StochasticBlock, layers=[200] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear', num_classes=num_classes)


# From "Deep Networks with Stochastic Depth" for SVHN Experiments
def ResNet152SVHN(num_classes=10):
    return ResNet(BasicBlock, layers=[25] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def StochasticResNet152SVHN(num_classes=10):
    return StochasticResNet(StochasticBlock, layers=[25] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear', num_classes=num_classes)


# Based on but not in "Deep Networks for Stochastic Depth"
def StochasticResNet56(num_classes=10):
    return StochasticResNet(StochasticBlock, layers=[9] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear', num_classes=num_classes)


def StochasticResNet56_08(num_classes=10):
    return StochasticResNet(StochasticBlock, layers=[9] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.8,
                            decay='linear', num_classes=num_classes)


# From "Wide Residual Networks"
def WRN(n, k, num_classes=10):
    assert (n - 4) % 6 == 0
    base_filters = [16, 32, 64]
    filters = [num_filters * k for num_filters in base_filters]
    d = (n - 4) / 2  # l = 2
    return ResNet(PreActBlock, layers=[int(d / 3)] * 3, filters=filters,
                  inplanes=16, num_classes=num_classes)


def WRN_40_4(num_classes=10):
    return WRN(40, 4, num_classes=num_classes)


def WRN_16_4(num_classes=10):
    return WRN(16, 4, num_classes=num_classes)


def WRN_16_8(num_classes=10):
    return WRN(16, 8, num_classes=num_classes)


def WRN_28_10(num_classes=10):
    return WRN(28, 10, num_classes=num_classes)


# From "Aggregated Residual Transformations for Deep Neural Networks"
def ResNeXt29(cardinality, base_width, num_classes=10):
    Block = partial(ResNeXtBottleneck, cardinality=cardinality,
                    base_width=base_width)
    Block.__name__ = ResNeXtBottleneck.__name__
    Block.expansion = ResNeXtBottleneck.expansion
    return ResNet(Block, layers=[3, 3, 3], filters=[64, 128, 256],
                  num_classes=num_classes)


# From kunagliu/pytorch
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck,
                  layers=[3, 4, 23, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck,
                  layers=[3, 8, 36, 3], filters=[64, 128, 256, 512])


MODELS = {
        # "Deep Residual Learning for Image Recognition"
        'resnet20': ResNet20,
        'resnet32': ResNet32,
        'resnet44': ResNet44,
        'resnet56': ResNet56,
        'resnet110': ResNet110,
        'resnet1202': ResNet1202,

        # "Wide Residual Networks"
        'wrn-40-4': WRN_40_4,
        'wrn-16-4': WRN_16_4,
        'wrn-16-8': WRN_16_8,
        'wrn-28-10': WRN_28_10,

        # Based on "Identity Mappings in Deep Residual Networks"
        'preact8': PreActResNet8,
        'preact14': PreActResNet14,
        'preact20': PreActResNet20,
        'preact56': PreActResNet56,
        'preact164-basic': PreActResNet164Basic,

        # "Identity Mappings in Deep Residual Networks"
        'preact110': PreActResNet110,
        'preact164': PreActResNet164,
        'preact1001': PreActResNet1001,

        # Based on "Deep Networks with Stochastic Depth"
        'stochastic56': StochasticResNet56,
        'stochastic56-08': StochasticResNet56_08,
        'stochastic110': StochasticResNet110,
        'stochastic1202': StochasticResNet1202,
        'stochastic152-svhn': StochasticResNet152SVHN,
        'resnet152-svhn': ResNet152SVHN,

        # "Aggregated Residual Transformations for Deep Neural Networks"
        'resnext29-8-64': lambda num_classes=10: ResNeXt29(8, 64, num_classes=num_classes),  # noqa: E501
        'resnext29-16-64': lambda num_classes=10: ResNeXt29(16, 64, num_classes=num_classes),  # noqa: E501

        # Kuangliu/pytorch-cifar
        'resnet18': ResNet18,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
}
