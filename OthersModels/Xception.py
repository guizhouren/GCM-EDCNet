from torch import nn
import torch
from torchscan import summary

# 定义深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, stride=1):
        super(SeparableConv2d, self).__init__()
        # 逐点卷积 pointwise, 1x1 卷积
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, groups=1)
        # 深度卷积 depthwise, 逐个通道操作, groups=in_channels=out_channels
        self.depth_conv = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, groups=out_ch)

    def forward(self, x):
        out = self.point_conv(x)
        out = self.depth_conv(out)
        return out


# 定义 带残差连接的深度可分离卷积模块_Entry部分
class ResDSC_Entry(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResDSC_Entry, self).__init__()
        self.residual = nn.Sequential(
                                      nn.ReLU(),
                                      SeparableConv2d(in_ch, out_ch, kernel_size=(3, 3), padding=1),
                                      nn.ReLU(),
                                      SeparableConv2d(out_ch, out_ch, kernel_size=(3, 3), padding=1),
                                      nn.MaxPool2d((3, 3), stride=(2, 2), padding=1))
        self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, (1, 1), stride=(2, 2)),
                                      )

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        output = shortcut + residual
        return output

class ResDSC_Middle(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResDSC_Middle, self).__init__()
        self.residual = nn.Sequential(
                                      nn.ReLU(),
                                      SeparableConv2d(in_ch, out_ch, kernel_size=(3, 3), padding=1),
                                      nn.ReLU(),
                                      SeparableConv2d(out_ch, out_ch, kernel_size=(3, 3), padding=1),
                                      nn.ReLU(),
                                      SeparableConv2d(out_ch, out_ch, kernel_size=(3, 3), padding=1),

                                      )

    def forward(self, x):
        residual = self.residual(x)
        output = x + residual
        return output


class Xception(nn.Module):
    def __init__(self,num_classes=14):
        super(Xception,self).__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1),
            ResDSC_Entry(64,128),
            ResDSC_Entry(128,256),
            ResDSC_Entry(256,728)
        )
        middle_layers = []
        for i in range(8):
            middle_layers.append(ResDSC_Middle(728,728))
        self.middle = nn.Sequential(
            *middle_layers
        )

        self.exit = nn.Sequential(
            ResDSC_Entry(728,1024),
            SeparableConv2d(1024,1536,kernel_size=3,padding=1),
            nn.ReLU(),
            SeparableConv2d(1536, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.linear = nn.Linear(2048,num_classes)

    def forward(self,x):
        res = self.entry(x)
        res = self.middle(res)
        res = self.exit(res)
        res = res.mean([2,3]) # global
        res = self.linear(res)
        return res


# X = torch.randn(2,3,224,224).cuda()
# net = Xception().cuda().eval()
# print(net(X))
# print(summary(net,(3,224,224)))