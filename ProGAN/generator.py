import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Progressive layers construction
        self.layerlist = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(512, 512, 4),
                    nn.LeakyReLU(0.2),
                    PixNorm(),
                    Eqlr_Conv(512, 512),
                    nn.LeakyReLU(0.2),
                    PixNorm(),
                ),
            ]
        )
        for i in range(3):
            self.layerlist.append(DoubleConv(512, 512))
        for i in range(3):
            self.layerlist.append(DoubleConv(2**(9-i), 2**(8-i)))

        # RGB layer construction
        self.RGBlayer = nn.ModuleList([Eqlr_Conv(512, 3, 1, 0)])
        for i in range(4):
            self.RGBlayer.append(Eqlr_Conv(512, 3, 1, 0))
        for i in range(3):
            self.RGBlayer.append(Eqlr_Conv(2**(8-i), 3, 1, 0))

        # Final Tanh
        self.final = nn.Tanh()

    def forward(self, x, phase: int, alpha: float):
        conved = self.layerlist[0](x)
        if phase == 0:
            conved_RGB = self.RGBlayer[0](conved)
            return self.final(conved_RGB)
        for step in range(phase):
            upsampled = nn.functional.interpolate(conved, scale_factor=2)
            conved = self.layerlist[step+1](upsampled)
        upsampled_RGB, conved_RGB = self.RGBlayer[phase](
            upsampled), self.RGBlayer[phase+1](conved)
        return self.final(interpolation(alpha, upsampled_RGB, conved_RGB))


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1, pixnorm=True, inv: bool = False) -> None:
        super().__init__()
        mid_channel = in_channel if inv else out_channel
        self.layerlist = nn.ModuleList([
            Eqlr_Conv(in_channel, mid_channel, 3, padding),
            Eqlr_Conv(mid_channel, out_channel, 3, padding)
        ])
        self.transform = PixNorm() if pixnorm else nn.Identity()

    def forward(self, x):
        for layer in self.layerlist:
            x = layer(x)
            x = self.transform(x)
        return x


class Eqlr_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernal_size=3, padding=1) -> None:
        super().__init__()
        self.scaler = (2/(in_channel * kernal_size * kernal_size)) ** 0.5
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernal_size, padding=padding)
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(self.scaler * x)
        return self.activation(x)


class PixNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def interpolation(alpha: float, upsampled: torch.Tensor, conved: torch.Tensor):
    return (alpha * conved) + ((1-alpha) * upsampled)


if __name__ == "__main__":
    g = Generator()
    x = torch.randn(4, 512, 1, 1)
    y = g(x, 4, 0.5)
    print(y.shape)
