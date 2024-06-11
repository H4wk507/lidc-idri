from torch import cat, nn
from torch.nn import functional as F


class Unet(nn.Module):
    def __init__(self, input_channel=1):
        super().__init__()
        self.inc = DoubleConv(input_channel, 64)

        self.encoder_1 = Encoder(64, 128)
        self.encoder_2 = Encoder(128, 256)
        self.encoder_3 = Encoder(256, 512)
        self.encoder_4 = Encoder(512, 1024)

        self.decoder_1 = Decoder(1024, 512)
        self.decoder_2 = Decoder(512, 256)
        self.decoder_3 = Decoder(256, 128)
        self.decoder_4 = Decoder(128, 64)

        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.encoder_1(x1)
        x3 = self.encoder_2(x2)
        x4 = self.encoder_3(x3)
        x5 = self.encoder_4(x4)

        x = self.decoder_1(x5, x4)
        x = self.decoder_2(x, x3)
        x = self.decoder_3(x, x2)
        x = self.decoder_4(x, x1)

        logits = self.outc(x)
        return logits

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(input_channel, out_channel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            input_channel, output_channel, stride=2, kernel_size=2
        )
        self.conv2d_1 = DoubleConv(input_channel, output_channel)

    def forward(self, x1, x2):
        x1 = self.conv_t(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = cat([x2, x1], dim=1)
        return self.conv2d_1(x)


