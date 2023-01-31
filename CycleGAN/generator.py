from torch import nn


class generator(nn.Module):
    def __init__(self):
        super().__init__()


class instance_relu(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class c7s1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 7,
                      padding=3, padding_mode="reflect"),
            instance_relu(out_channel)
        )

    def forward(self, x):
        return self.layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3,
                      padding=1, padding_mode="reflect"),
            nn.Conv2d(out_channel, out_channel, 3,
                      padding=1, padding_mode="reflect"),
        )

    def forward(self, x):
        return x + self.layer(x)


class c3s2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2,
                      1, padding_mode="reflect",),
            instance_relu(out_channel)
        )


class convT(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 3,
                               2, 1, padding_mode="reflect"),
            instance_relu(out_channel),
        )

    def forward(self, x):
        return self.layer(x)
