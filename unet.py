from torch import cat, nn


class Unet(nn.Module):
    def __init__(self, input_channel=1):
        super().__init__()
        self.encoder_1 = Encoder(input_channel, 64)
        self.encoder_2 = Encoder(64, 128)
        self.encoder_3 = Encoder(128, 256)
        self.encoder_4 = Encoder(256, 512)

        self.conv_block = ConvBlock(512, 1024)

        self.decoder_1 = Decoder(1024, 512)
        self.decoder_2 = Decoder(512, 256)
        self.decoder_3 = Decoder(256, 128)
        self.decoder_4 = Decoder(128, 64)

        self.cls = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """------ Encoder ------"""
        x1, p1 = self.encoder_1(x)
        x2, p2 = self.encoder_2(p1)
        x3, p3 = self.encoder_3(p2)
        x4, p4 = self.encoder_4(p3)

        """ ------ BottleNeck ------"""
        x5 = self.conv_block(p4)

        """ ------ Decoder ------"""
        x6 = self.decoder_1(x5, x4)
        x7 = self.decoder_2(x6, x3)
        x8 = self.decoder_3(x7, x2)
        x9 = self.decoder_4(x8, x1)

        """ ------ Final Layer ------"""
        x_final = self.cls(x9)
        x_final = self.act(x_final)

        return x_final

class Encoder(nn.Module):
    def __init__(self, input_channel, out_channel, dropout=0.5):
        super(Encoder, self).__init__()
        self.conv2d_1 = ConvBlock(input_channel, out_channel, dropout)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv2d_1(x)
        p = self.maxpool(x)
        p = self.dropout(p)

        return x, p

class ConvBlock(nn.Module):
    def __init__(self, input_channel, out_channel, dropout=0.5):
        super(ConvBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(input_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.conv2d_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.batchnorm_1(x)
        x = self.act(x)

        x = self.dropout(x)

        x = self.conv2d_2(x)
        x = self.batchnorm_2(x)
        x = self.act(x)

        return x


class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel, dropout=0.5):
        super(Decoder, self).__init__()
        self.conv_t = nn.ConvTranspose2d(
            input_channel, output_channel, stride=2, kernel_size=2
        )
        self.conv2d_1 = ConvBlock(output_channel * 2, output_channel, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip):
        x = self.conv_t(x)
        x = cat([x, skip], dim=1)
        x = self.dropout(x)
        x = self.conv2d_1(x)

        return x


