import torch
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
import torch.nn as nn


def conv3x3x3(in_channels: int, out_channels: int, stride: int=1, dilation:int=1, bias=False):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_channels, 
                     out_channels, 
                     kernel_size=3, 
                     stride=stride, 
                     padding=dilation, 
                     dilation=dilation, 
                     bias=bias)

def conv1x1x1(in_channels: int, out_channels: int, stride=1, dilation:int=1, bias=False):
    return nn.Conv3d(in_channels, 
                     out_channels, 
                     kernel_size=1, 
                     stride=stride,
                     dilation=dilation, 
                     bias=bias)

class BasicBlock(nn.Module):
    extension : int = 1

    def __init__(
        self, 
        in_channels,
        out_channels,
        stride = 1,
        base_width : int = 64,
        groups : int = 1,
        dilation : int = 1,
        norm_layer = None,
        downsample = None
    ) -> None:
        super().__init__()

        if groups != 1 or base_width != 64:
            raise("BasicBlock can't have the groups != 1 and base_width > 64")
        
        if dilation != 1:
            raise("dilation can't != 1")
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self.Conv1 = conv3x3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.Conv2 = conv3x3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.Conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.Conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    extension = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        downsample = None,
        groups : int = 1,
        base_width : int = 64,
        dilation : int = 1,
        norm_layer = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(in_channels * (base_width / 64.0)) * groups
        self.Conv1 = conv1x1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.Conv2 = conv3x3x3(width, out_channels, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.Conv3 = conv1x1x1(width, out_channels * self.extension)
        self.bn3 = norm_layer(out_channels * self.extension)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x : Tensor) -> Tensor:
        identity = x

        out = self.Conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.Conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.Conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out




class Resnet(nn.Module):
    def __init__(self, block : Union[BasicBlock, Bottleneck], layers: List=[], num_classes: int=512, groups: int=1, width_per_group: int =64, replace_stride_with_dilation: Optional[List[bool]]=None ,base_width: int=64, norm_layer: Optional[Callable[..., nn.Module]]=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.in_channels = 64
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(1, self.in_channels, kernel_size=5, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = norm_layer(self.in_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2)
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

        self.fc = nn.Linear(512 * block.extension, num_classes)
    
    def _make_layer(self, block, blocks, channels, stride: int=1, dilate: bool=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if self.in_channels != channels * block.extension or stride != 1:
            downsample = nn.Sequential(
                conv1x1x1(self.in_channels, channels * block.extension, stride),
                norm_layer( channels * block.extension)
            )

        layers = []
        layers.append(
            #
            block(self.in_channels, channels, stride=stride, downsample=downsample, dilation=previous_dilation, groups=self.groups, norm_layer=norm_layer, base_width=self.base_width)
        )
        self.in_channels = channels * block.extension
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, channels, groups=self.groups, norm_layer=norm_layer, dilation=self.dilation, base_width=self.base_width)
            )
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def resnet10(num_classes=512):
    return Resnet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=num_classes)
def resnet18(num_classes=512):
    return Resnet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)

if __name__ == "__main__":
    array = torch.randn(1, 1, 256, 256, 256)
    a1 = torch.ones(1, 1, 32,32, 32)
    a2 = torch.ones(1, 1, 28,32, 32)
    array = torch.tensor()
    model = resnet18()
    print(model)
    output = model(array)
    print(output.shape)
    
        
        