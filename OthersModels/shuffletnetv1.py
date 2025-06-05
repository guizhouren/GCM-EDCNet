from torch import nn
import torch.nn.functional as F
import torch
from torchscan import summary
def shuffle(x, groups):
    N, C, H, W = x.size()
    out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):# 输入通道数,输出通道数,步长,分组数
        super().__init__()

        mid_channles = int(out_channels / 4) # out_channels通道数为中间通道数的4倍

        if in_channels == 24: # 如果输入通道数为24则分组数为1
            self.groups = 1
        else:
            self.groups = groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channles, 1, groups=self.groups, bias=False), #进行分组卷积
            nn.BatchNorm2d(mid_channles),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channles, mid_channles, 3, stride=stride, padding=1, groups=mid_channles, bias=False),# 进行深度卷积
            nn.BatchNorm2d(mid_channles),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channles, out_channels, 1, groups=groups, bias=False),# 进行点卷积
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1)) # 多尺度,当stride=2时使用

        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.stride == 2: # 如果步长为二
            res = self.shortcut(x) # avgpool2d操作
            out = F.relu(torch.cat([out,res], 1)) # 拼接两条线的操作结果
        else:# 如果步长为2
            out = F.relu(out + x)# 执行相加操作

        return out


class ShuffleNet(nn.Module):
    def __init__(self, groups, channel_num, class_num=14):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self.make_layers(24, channel_num[0], 4, 2, groups)
        self.stage3 = self.make_layers(channel_num[0], channel_num[1], 8, 2, groups)
        self.stage4 = self.make_layers(channel_num[1], channel_num[2], 4, 2, groups)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel_num[2], class_num)

    def make_layers(self, input_channels, output_channels, layers_num, stride, groups):
        layers = []
        layers.append(Bottleneck(input_channels, output_channels - input_channels, stride, groups)) # 每个stage中的第一个卷积层的输出层通道数为out减去in
        input_channels = output_channels

        for i in range(layers_num - 1):
            Bottleneck(input_channels, output_channels, 1, groups) # 输入层与输出层的通道数相等

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


# mod = ShuffleNet(3,[240,480,960],14)
# print(summary(mod,(3,224,224)))