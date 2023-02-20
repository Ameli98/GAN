import torch
from torch import nn, Tensor
from generator import DoubleConv, Eqlr_Conv, interpolation


def minibatch(input: Tensor) -> Tensor:
    constant = torch.std(input, dim=0).mean().repeat(
        input.shape[0], 1, input.shape[2], input.shape[3])
    output = torch.cat((input, constant), dim=1)
    return output


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # progresivce layer
        self.layerlist = nn.ModuleList([])
        for i in range(3):
            self.layerlist.append(DoubleConv(
                2**(i+6), 2**(i+7), pixnorm=False, inv=True))
        for i in range(3):
            self.layerlist.append(DoubleConv(
                512, 512, pixnorm=False, inv=True))

        # RGB layer
        self.RGBlayer = nn.ModuleList([Eqlr_Conv(3, 512, 1, 0)])
        for i in range(3):
            self.RGBlayer.append(Eqlr_Conv(3, 2**(6+i), 1, 0))
        for i in range(3):
            self.RGBlayer.append(Eqlr_Conv(3, 512, 1, 0))

        # final layer
        self.finallayer = nn.Sequential(
            Eqlr_Conv(513, 512),
            nn.LeakyReLU(0.2),
            Eqlr_Conv(512, 512, 4, 0),
            nn.Conv2d(512, 1, 1)
        )

    # Binding final layer with minibatch
    def ending(self, input: Tensor) -> int:
        output = minibatch(input)
        output = self.finallayer(output)
        return output

    def forward(self, input: Tensor, phase: int, alpha: float):
        if phase == 0:
            input = self.RGBlayer[0](input)
            return self.ending(input)
        pool = nn.AvgPool2d(2)
        downsampled = pool(input)
        downsampled_nRGB = self.RGBlayer[7-phase](downsampled)
        conved_nRGB = self.RGBlayer[6-phase](input)
        conved_nRGB = pool(self.layerlist[5-phase](conved_nRGB))

        output = interpolation(alpha, downsampled_nRGB, conved_nRGB)
        for layer in self.layerlist[7-phase:]:
            output = layer(output)
            output = pool(output)
        return self.ending(output)


if __name__ == "__main__":
    d = Discriminator()
    x = torch.randn(4, 3, 256, 256)
    y = d(x, 6, 1)
    print(y.shape)
