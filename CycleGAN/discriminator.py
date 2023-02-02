from torch import nn


class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 4, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            c4s2(64, 128),
            c4s2(128, 256),
            c4s2(256, 512),
        )
        for m in self.modules():
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        return self.layer(x)


class c4s2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4,
                      padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layer(x)
