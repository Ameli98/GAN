from torch import nn

# TODO: fix discriminator's outpus size to [5, 1, 30, 30]


class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 4, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            c4s2(64, 128),
            c4s2(128, 256),
            c4s2(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1, padding_mode="reflect")
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        return nn.Sigmoid(self.layer(x))


class c4s2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, 2),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layer(x)


if __name__ == "__main__":
    import torch
    x = torch.randn((5, 3, 256, 256))
    model = discriminator()
    preds = model(x)
    print(preds.shape)
