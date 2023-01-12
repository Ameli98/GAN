import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, image_channels, features_d, image_size, num_classes):
        super().__init__()
        self.layer = nn.Sequential(
            self._block(image_channels+1, features_d),
            self._block(features_d, features_d*2),
            self._block(features_d*2, features_d*4),
            self._block(features_d*4, features_d*8),
            nn.Conv2d(features_d*8, 1, 4, 2, 0),
        )
        initial_normalize(self)

        self.image_size = image_size
        self.emd = nn.Embedding(num_classes, image_size * image_size)

    def _block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, label):
        embedding = self.emd(label).view(
            label.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat([x, embedding], dim=1)
        return self.layer(x)


class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, emd_size, image_channels, features_g):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(noise_dim+emd_size, features_g*16, 4, 1, 0),
            nn.ReLU(),
            self._block(features_g*16, features_g*8),
            self._block(features_g*8, features_g*4),
            self._block(features_g*4, features_g*2),
            nn.ConvTranspose2d(features_g*2, image_channels, 4, 2, 1),
            nn.Tanh(),
        )
        initial_normalize(self)

        self.emd = nn.Embedding(num_classes, emd_size)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x, label):
        embedding = self.emd(label).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.layer(x)


def initial_normalize(Model):
    for m in Model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, std=0.02)


def gradient_penalty(discriminator: Critic, label, real, fake, device="cpu"):
    batch, C, H, W = real.shape
    weight = torch.rand(batch, 1, 1, 1).repeat(1, C, H, W).to(device)
    interpolated_image = real * weight + fake * (1 - weight)
    mix_score = discriminator(interpolated_image, label)
    gradient = torch.autograd.grad(
        outputs=mix_score, inputs=interpolated_image,
        grad_outputs=torch.ones_like(mix_score),
        retain_graph=True, create_graph=True)[0]
    grad_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((grad_norm - 1) ** 2)
    return penalty
