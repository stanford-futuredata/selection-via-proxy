from collections import OrderedDict
from typing import Mapping, Optional
# from typing import Protocol  # Python 3.8 and above
from typing_extensions import Protocol

from torch import nn
from torch.nn import functional as F


def kmax_pooling(inputs, dim, k):
    indices = (inputs
               .topk(k, dim=dim)[1]  # indices of topk
               .sort(dim=dim)[0])    # preserve original order
    return inputs.gather(dim, indices)


class KMaxPool1d(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, inputs):
        return kmax_pooling(inputs, 2, self.k)

    def __repr__(self):
        fmt_str = self.__class__.__name__
        fmt_str += "(k={0})".format(self.k)
        return fmt_str


class ConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_filters, out_filters, shortcut=False, bias=False):
        super().__init__()
        self.shortcut = shortcut

        if out_filters > in_filters:
            stride = 2
        else:
            stride = 1

        self.conv1 = nn.Conv1d(in_filters, out_filters, 3, padding=1,
                               stride=stride, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = nn.Conv1d(out_filters, out_filters, 3, padding=1,
                               stride=1, bias=bias)
        self.bn2 = nn.BatchNorm1d(out_filters)

        if self.shortcut:
            if stride != 1:
                self._shortcut = nn.Sequential(
                    nn.Conv1d(in_filters, out_filters, 1,
                              stride=stride, bias=bias)
                )
            else:
                self._shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)

        if self.shortcut:
            H += self._shortcut(inputs)

        H = F.relu(H)

        return H


class MaxPoolBlock(nn.Module):
    expansion = 1

    def __init__(self, in_filters, out_filters, shortcut=False, bias=False):
        super().__init__()
        self.shortcut = shortcut

        self.increasing = out_filters > in_filters
        if self.increasing:
            self.max_pool = nn.MaxPool1d(3, stride=2, padding=1)

        self.conv1 = nn.Conv1d(in_filters, out_filters, 3, padding=1,
                               stride=1, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = nn.Conv1d(out_filters, out_filters, 3, padding=1,
                               stride=1, bias=bias)
        self.bn2 = nn.BatchNorm1d(out_filters)

        if self.shortcut:
            if self.increasing:
                self._shortcut = nn.Sequential(
                    nn.Conv1d(in_filters, out_filters, 1, stride=1, bias=bias)
                )
            else:
                self._shortcut = nn.Sequential()

    def forward(self, inputs):
        if self.increasing:
            inputs = self.max_pool(inputs)

        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)

        if self.shortcut:
            H += self._shortcut(inputs)
        H = F.relu(H)
        return H


class KMaxPoolBlock(nn.Module):
    expansion = 1

    def __init__(self, in_filters, out_filters, shortcut=False, bias=False):
        super().__init__()
        self.shortcut = shortcut
        self.increasing = out_filters > in_filters

        self.conv1 = nn.Conv1d(in_filters, out_filters, 3, padding=1,
                               stride=1, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = nn.Conv1d(out_filters, out_filters, 3, padding=1,
                               stride=1, bias=bias)
        self.bn2 = nn.BatchNorm1d(out_filters)

        if self.shortcut:
            if self.increasing:
                self._shortcut = nn.Sequential(
                    nn.Conv1d(in_filters, out_filters, 1, stride=1, bias=bias)
                )
            else:
                self._shortcut = nn.Sequential()

    def forward(self, inputs):
        if self.increasing:
            inputs = kmax_pooling(inputs, 2, inputs.size(2) // 2)

        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)

        if self.shortcut:
            H += self._shortcut(inputs)
        H = F.relu(H)
        return H


class VDCNN(nn.Module):

    def __init__(self, Block, blocks, filters,
                 vocab_size=69, embedding_size=16, k=8, num_hidden=2048,
                 num_classes=5, in_filters=None, shortcut=False,
                 bias=False):

        in_filters = in_filters or filters[0]

        super().__init__()

        self.character_embedding = nn.Embedding(
            vocab_size, embedding_size,
            scale_grad_by_freq=False,  # Not Sure
            sparse=False)
        self.conv1 = nn.Conv1d(embedding_size, in_filters, 3,
                               padding=1, stride=1, bias=bias)

        sections = OrderedDict()
        for section_index, (num_blocks, out_filters) in enumerate(zip(blocks, filters)):  # noqa: E501
            section = OrderedDict()
            for block_index in range(num_blocks):
                block = Block(in_filters, out_filters,
                              shortcut=shortcut, bias=bias)
                section["block_{}".format(block_index)] = block
                in_filters = out_filters * Block.expansion

            sections["section_{}".format(section_index)] = nn.Sequential(section)  # noqa: E501
        self.sections = nn.Sequential(sections)
        self.kmax = KMaxPool1d(k)

        self.fc1 = nn.Linear(in_filters * k, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # inputs.shape -> (batch_size, max_length)
        H = self.character_embedding(inputs)
        # H.shape -> (batch_size, max_length, embedding_size)
        H = H.transpose(1, 2)
        # H.shape -> (batch_size, embedding_size, max_length)
        H = self.conv1(H)
        # H.shape -> (batch_size, filters[0], max_length)

        H = self.sections(H)

        H = self.kmax(H)
        H = H.view(H.size(0), -1)
        H = self.fc1(H)
        H = F.relu(H)
        H = self.fc2(H)
        H = F.relu(H)
        H = self.fc3(H)

        return H


def VDCNN9Conv(num_classes=5, shortcut=False, bias=False):
    return VDCNN(ConvBlock, blocks=[1, 1, 1, 1], filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN9MaxPool(num_classes=5, shortcut=False, bias=False):
    return VDCNN(MaxPoolBlock, blocks=[1, 1, 1, 1],
                 filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN9MaxPoolLite(num_classes=5, shortcut=False, bias=False):
    return VDCNN(MaxPoolBlock, blocks=[1, 1, 1, 1],
                 filters=[16, 32, 64, 128],
                 num_hidden=512,
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN9KMaxPool(num_classes=5, shortcut=False, bias=False):
    return VDCNN(KMaxPoolBlock, blocks=[1, 1, 1, 1],
                 filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN17Conv(num_classes=5, shortcut=False, bias=False):
    return VDCNN(ConvBlock, blocks=[2, 2, 2, 2], filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN17MaxPool(num_classes=5, shortcut=False, bias=False):
    return VDCNN(MaxPoolBlock, blocks=[2, 2, 2, 2],
                 filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN17KMaxPool(num_classes=5, shortcut=False, bias=False):
    return VDCNN(KMaxPoolBlock, blocks=[2, 2, 2, 2],
                 filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN29Conv(num_classes=5, shortcut=False, bias=False):
    return VDCNN(ConvBlock, blocks=[5, 5, 2, 2], filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN29MaxPool(num_classes=5, shortcut=False, bias=False):
    return VDCNN(MaxPoolBlock, blocks=[5, 5, 2, 2],
                 filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN29KMaxPool(num_classes=5, shortcut=False, bias=False):
    return VDCNN(KMaxPoolBlock, blocks=[5, 5, 2, 2],
                 filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN49Conv(num_classes=5, shortcut=False, bias=False):
    return VDCNN(ConvBlock, blocks=[8, 8, 5, 3], filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN49MaxPool(num_classes=5, shortcut=False, bias=False):
    return VDCNN(MaxPoolBlock, blocks=[8, 8, 5, 3],
                 filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


def VDCNN49KMaxPool(num_classes=5, shortcut=False, bias=False):
    return VDCNN(KMaxPoolBlock, blocks=[8, 8, 5, 3],
                 filters=[64, 128, 256, 512],
                 num_classes=num_classes, shortcut=shortcut, bias=bias)


class ModelBuilder(Protocol):
    def __call__(self, num_classes: Optional[int]) -> nn.Module:
        pass


MODELS: Mapping[str, ModelBuilder] = {
    'vdcnn9-conv': VDCNN9Conv,
    'vdcnn9-conv-shortcut': lambda num_classes=5: VDCNN9Conv(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn9-maxpool': VDCNN9MaxPool,
    'vdcnn9-maxpool-lite': VDCNN9MaxPoolLite,
    'vdcnn9-maxpool-shortcut': lambda num_classes=5: VDCNN9MaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn9-kmaxpool': VDCNN9KMaxPool,
    'vdcnn9-kmaxpool-shortcut': lambda num_classes=5: VDCNN9KMaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501

    'vdcnn17-conv': VDCNN17Conv,
    'vdcnn17-conv-shortcut': lambda num_classes=5: VDCNN17Conv(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn17-maxpool': VDCNN17MaxPool,
    'vdcnn17-maxpool-shortcut': lambda num_classes=5: VDCNN17MaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn17-kmaxpool': VDCNN17KMaxPool,
    'vdcnn17-kmaxpool-shortcut': lambda num_classes=5: VDCNN17KMaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501

    'vdcnn29-conv': VDCNN29Conv,
    'vdcnn29-conv-shortcut': lambda num_classes=5: VDCNN29Conv(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn29-maxpool': VDCNN29MaxPool,
    'vdcnn29-maxpool-shortcut': lambda num_classes=5: VDCNN29MaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn29-kmaxpool': VDCNN29KMaxPool,
    'vdcnn29-kmaxpool-shortcut': lambda num_classes=5: VDCNN29KMaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501

    'vdcnn49-conv': VDCNN49Conv,
    'vdcnn49-conv-shortcut': lambda num_classes=5: VDCNN49Conv(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn49-maxpool': VDCNN49MaxPool,
    'vdcnn49-maxpool-shortcut': lambda num_classes=5: VDCNN49MaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn49-kmaxpool': VDCNN49KMaxPool,
    'vdcnn49-kmaxpool-shortcut': lambda num_classes=5: VDCNN49KMaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501
}
