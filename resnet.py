import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPoolShortCut(nn.Module):
    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(
            x.shape[0],
            self.out_c - self.in_c,
            x.shape[2],
            x.shape[3],
            device=x.device,
        )
        x = torch.cat((x, pad), dim=1)
        return x



class AvgPoolShortCut(nn.Module):
    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(
            x.shape[0],
            self.out_c - self.in_c,
            x.shape[2],
            x.shape[3],
            device=x.device,
        )
        x = torch.cat((x, pad), dim=1)
        return x

class BasicBlockCIFAR(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_type="conv"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if down_type == "conv":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            elif down_type == "ddu":
                self.shortcut = nn.Sequential(
                    AvgPoolShortCut(
                        stride=stride, out_c=self.expansion * planes, in_c=in_planes
                    )
                )
            else:
                raise ValueError("Invalid downsample type provided.")

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class BottleneckCIFAR(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_type="conv"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.act3 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if down_type == "conv":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            elif down_type == "ddu":
                self.shortcut = nn.Sequential(
                    AvgPoolShortCut(
                        stride=stride, out_c=self.expansion * planes, in_c=in_planes
                    )
                )
            else:
                raise ValueError("Invalid downsample type provided.")

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        block,
        depth,
        width_multiplier=1,
        num_classes=10,
        down_type="conv",
        act="relu",
    ):
        super().__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        self.num_features = 64 * block.expansion * width_multiplier
        self.down_type = down_type

        assert (
            depth - 2
        ) % 6 == 0, "depth should be 6n+2 (e.g., 20, 32, 44, 56, 110, 1202)"
        n = (depth - 2) // 6

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU() if act == "relu" else nn.LeakyReLU()
        self.layer1 = self.make_layer(block, 16 * width_multiplier, n, stride=1)
        self.layer2 = self.make_layer(block, 32 * width_multiplier, n, stride=2)
        self.layer3 = self.make_layer(block, 64 * width_multiplier, n, stride=2)
        self.global_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def make_layer(self, block, planes, num_blocks, stride):
        blocks = [
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                down_type=self.down_type,
            )
        ]
        self.in_planes = planes * block.expansion

        for _ in range(num_blocks - 1):
            blocks.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=1,
                    down_type=self.down_type,
                )
            )

        return nn.Sequential(*blocks)

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def forward_features(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out

    def forward_head(self, x, pre_logits: bool = False):
        out = self.global_pool(x)
        out = out.view(out.size(0), -1)

        return out if pre_logits else self.fc(out)

    def forward(self, x):
        out = self.forward_features(x)
        out = self.forward_head(out)

        return out


def make_resnet_cifar(depth, num_classes=10, down_type = "conv", act="relu"):
    return ResNetCIFAR(
        block=BasicBlockCIFAR,
        depth=depth,
        num_classes=num_classes,
        down_type=down_type,
        act=act,
    )
