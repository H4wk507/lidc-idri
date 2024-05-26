import torch
from torch import nn


class ResUNet(nn.Module):
    def __init__(self, in_channels: int, channels: list[int]):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
        )

        self.squeeze_excitation1 = SqueezeExcitation(channels[0], channels[0] // 2)
        self.res_conv1 = ResidualConv(channels[0], channels[1], 2)

        self.squeeze_excitation2 = SqueezeExcitation(channels[1], channels[1] // 2)
        self.rev_conv2 = ResidualConv(channels[1], channels[2], 2)

        self.squeeze_excitation3 = SqueezeExcitation(channels[2], channels[2] // 2)
        self.rev_conv3 = ResidualConv(channels[2], channels[3], 2)

        self.aspp_bridge = ASPP(channels[3], channels[4])

        self.attention1 = AttentionBlock(channels[2], channels[4])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_rev_conv1 = ResidualConv(channels[4] + channels[2], channels[3])

        self.attention2 = AttentionBlock(channels[1], channels[3])
        self.up_rev_conv2 = ResidualConv(channels[3] +  channels[1], channels[2])

        self.attention3 = AttentionBlock(channels[0], channels[2])
        self.up_rev_conv3 = ResidualConv(channels[2] + channels[0], channels[1])

        self.out_layer = nn.Sequential(
            ASPP(channels[1], channels[0]),
            nn.Conv2d(channels[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x1 = self.input_layer(x) + x

        x2 = self.squeeze_excitation1(x1)
        x2 = self.res_conv1(x2)

        x3 = self.squeeze_excitation2(x2)
        x3 = self.rev_conv2(x3)

        x4 = self.squeeze_excitation3(x3)
        x4 = self.rev_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attention1(x3, x5)
        x6 = self.upsample(x6)
        x6 = torch.cat([x3, x6], dim=1)
        x6 = self.up_rev_conv1(x6)

        x7 = self.attention2(x2, x6)
        x7 = self.upsample(x7)
        x7 = torch.cat([x2, x7], dim=1)
        x7 = self.up_rev_conv2(x7)

        x8 = self.attention3(x1, x7)
        x8 = self.upsample(x8)
        x8 = torch.cat([x1, x8], dim=1)
        x8 = self.up_rev_conv3(x8)

        out = self.out_layer(x8)
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()

        self.model = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(squeeze_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = nn.AdaptiveAvgPool2d(1)(x)
        y = self.model(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.model(x) + self.conv_skip(x)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dilations = [1, 6, 12, 18]

        self.aspp_block1 = ASPPBlock(in_channels, out_channels, 1, 0, self.dilations[0])
        self.aspp_block2 = ASPPBlock(in_channels, out_channels, 3, self.dilations[1], self.dilations[1])
        self.aspp_block3 = ASPPBlock(in_channels, out_channels, 3, self.dilations[2], self.dilations[2])
        self.aspp_block4 = ASPPBlock(in_channels, out_channels, 3, self.dilations[3], self.dilations[3])

        self.out = nn.Conv2d(len(self.dilations) * out_channels, out_channels, kernel_size=1)
        self.init_weight()
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        x4 = self.aspp_block4(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.out(x)

        return x


class ASPPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, dilation: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.model(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.attention = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, 1)
        )

    def forward(self, x1, x2):
        out = self.encoder(x1) + self.decoder(x2)
        out = self.attention(out)
        return out * x2
