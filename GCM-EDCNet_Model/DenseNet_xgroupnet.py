import torch
from torch import nn


def shuffle(x, groups):
    N, C, H, W = x.size()
    out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    return out

class Xgroup_block(nn.Module):
    def __init__(self,input_channels,num_channels,ratio=4,group=4): #ratio=4
        super().__init__()
        self.group=group
        self.expand_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels * ratio, kernel_size=1, groups=self.group),
            nn.BatchNorm2d(input_channels * ratio), nn.ReLU(),
        )
        self.dw_conv = nn.Conv2d(input_channels*ratio,input_channels*ratio,kernel_size=3,padding=1,groups=input_channels*ratio)
        self.squeeze_conv = nn.Sequential(
            nn.Conv2d(input_channels*ratio, num_channels, kernel_size=1, groups=self.group),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self,X):
        X = self.expand_conv(X)
        X = shuffle(X,groups=self.group)
        X = self.dw_conv(X)
        X = self.squeeze_conv(X)
        X = shuffle(X,groups=self.group)
        return X


def conv_block(input_channels,num_channels,ratio=4): #ratio=4
    # return nn.Sequential(
    #     nn.Conv2d(input_channels,input_channels*ratio,kernel_size=1,groups=3),
    #     nn.BatchNorm2d(input_channels*ratio),nn.ReLU(),
    #     nn.Conv2d(input_channels*ratio,input_channels*ratio,kernel_size=3,padding=1,groups=input_channels*ratio),
    #     nn.BatchNorm2d(input_channels*ratio),
    #     nn.Conv2d(input_channels, input_channels * ratio, kernel_size=1,groups=3)
    # )
    block = Xgroup_block(input_channels,num_channels,ratio)
    return block


class DenseBlock(nn.Module):
    # 稠密块(Dense block)
    def __init__(self,num_convs,input_channels,num_channels):
        super(DenseBlock,self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels*i+input_channels,num_channels
            ))
        self.net = nn.Sequential(*layer)

    def forward(self,X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个卷积块的输入和输出(Connect the inputs and outputs of each convolutional block in the channel dimension)
            X = torch.cat((X,Y),dim=1)
        return X



def transition_block(input_channels,num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )


b1 = nn.Sequential(
    nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
    nn.BatchNorm2d(64),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

# num_channels为当前的通道数
num_channels,growth_rate = 64,32
num_convs_in_dense_blocks = [4,4,4,4]
blks = []
for i,num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs,num_channels,growth_rate))
    # 上一个稠密快的输出通道数
    num_channels += num_convs*growth_rate
    # 在稠密块之间添加一个过渡层，使得通道数减半
    if i !=len(num_convs_in_dense_blocks)-1:
        blks.append(transition_block(num_channels,num_channels//2))
        num_channels = num_channels//2


def dense_xgroupnet(num_classes=4):
    return nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, num_classes)
    )