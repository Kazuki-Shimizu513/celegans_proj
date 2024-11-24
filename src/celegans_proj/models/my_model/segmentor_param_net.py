
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union


from torch import Tensor
import torch as th
from torch import nn
from torch.nn import functional as F

class MyDiffSegParamNet(nn.Module):
    def __init__(
        self,
        in_channel = 64,
        KL_len = 4,
    ):
        super().__init__()
        self.layer1 = Bottleneck(in_channel, 64,)
        self.layer2 = Bottleneck(64, 128,)
        self.layer3 = Bottleneck(128, 256,)
        self.layer4 = Bottleneck(256, 512,)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512,out_features=KL_len)
        self.last_non_linearity = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: Tensor) -> Tensor:

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = th.flatten(x, 1)
        x = self.fc(x)
        x = self.last_non_linearity(x)
        return x

class Bottleneck(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act_fn: str = "SiLU",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        if act_fn=="SiLU" or act_fn=="swish":
            self.non_linearity = nn.SiLU(inplace=True)
        elif act_fn=="ReLU":
            self.non_linearity = nn.ReLU(inplace=True)

        self.downsample =  nn.Sequential(
            conv1x1(inplanes, planes* self.expansion, stride),
            norm_layer(planes* self.expansion),
        )

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.non_linearity(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.non_linearity(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.non_linearity(out)

        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


if __name__ == "__main__":
    B = 2
    C = 64
    W = H = 64
    latents = th.randn(B,C,W,H)

    model = MyDiffSegParamNet(in_channel=64, KL_len=4)
    KL_THRESHOLD = model(latents)
    
    print(KL_THRESHOLD.shape)
    print(KL_THRESHOLD)

