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
        for i in range(5):
            self.layerlist.append(DoubleConv(2**(9-i), 2**(8-i)))

        # RGB layer construction
        self.RGBlayer = nn.ModuleList([])
        for i in range(4):
            self.RGBlayer.append(Eqlr_Conv(512, 3, 1))
        for i in range(4):
            self.RGBlayer.append(Eqlr_Conv(2**(8-i), 3, 1))

    def interpolation(self, alpha:float, upsampled:torch.Tensor, conved:torch.Tensor):
        return (alpha * conved) + ((1-alpha) * upsampled)

    def forward(self, x, phase, alpha):
        conved = self.layerlist[0](x)
        if phase == 0:
            return nn.Tanh(conved)
        if alpha == 1:
            for layer in self.layerlist[1:phase-1]:
                conved = layer(conved)
            return nn.Tanh(conved)
        for step in range(phase):
            upsampled = nn.functional.interpolate(conved, scale_factor=2)
            conved = self.layerlist[step+1](upsampled)
        upsampled_RGB, conved_RGB = self.RGBlayer[phase-1], self.RGBlayer[phase]
        return nn.Tanh(self.interpolation(alpha, upsampled_RGB, conved_RGB))


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, pixnorm=True) -> None:
        super().__init__()
        self.layerlist = nn.ModuleList([
            Eqlr_Conv(in_channel, out_channel, 3),
            Eqlr_Conv(in_channel, out_channel, 3)
        ])
        self.pixnorm = pixnorm
        

    def forward(self, x):
        for layer in self.layerlist:
            x = layer(x)
            x =  PixNorm(x) if self.pixnorm else x
        return x

class Eqlr_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernal_size=3) -> None:
        super().__init__()
        self.scaler = (2/(in_channel * kernal_size * kernal_size)) ** 0.5
        self.conv = nn.Conv2d(in_channel, out_channel, kernal_size)
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
        return torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True))