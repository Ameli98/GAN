import torch
import torch.nn as nn
import torchvision as vi
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, image_channels, features_d):
        super().__init__()
        self.layer = nn.Sequential(
            self._block(image_channels, features_d),
            self._block(features_d, features_d*2),
            self._block(features_d*2, features_d*4),
            self._block(features_d*4, features_d*8),
            nn.Conv2d(features_d*8, 1, 4, 2, 0),
            nn.Sigmoid(),
        )
        initial_normalize(self)

    def _block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layer(x)


class Generator(nn.Module):
    def __init__(self, noise_dim, image_channels, features_g):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, features_g*16, 4, 1, 0),
            nn.ReLU(),
            self._block(features_g*16, features_g*8),
            self._block(features_g*8, features_g*4),
            self._block(features_g*4, features_g*2),
            nn.ConvTranspose2d(features_g*2, image_channels, 4, 2, 1),
            nn.Tanh(),
        )
        initial_normalize(self)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


def initial_normalize(Model):
    for m in Model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, std=0.02)


if __name__ == "__main__":
    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)
    epochs, batch_size, lr = 5, 128, 2e-4
    image_size, image_channels = 64, 1
    noise_dim, features_d, features_g = 100, 128, 64
    transforms = vi.transforms.Compose(
        [
            vi.transforms.Resize((image_size, image_size)),
            vi.transforms.ToTensor(),
            vi.transforms.Normalize(
                [0.5 for channel in range(image_channels)],
                [0.5 for channel in range(image_channels)]
            ),
        ]
    )

    # Generator & Discriminator
    disc = Discriminator(image_channels, features_d).to(device)
    gen = Generator(noise_dim, image_channels, features_g).to(device)

    # Dataset & Dataloader
    dataset = vi.datasets.MNIST(
        root="dataset/",
        transform=transforms,
        download=True
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Optimizer & loss function
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    loss = nn.BCELoss()

    # SummaryWritter
    real_writter = SummaryWriter(f"log/real")
    fake_writter = SummaryWriter(f"log/fake")
    test_noise = torch.randn(32, noise_dim, 1, 1).to(device)
    step = 0

    # Trainning loop
    for epoch in range(epochs):
        for batch_index, (real, label) in enumerate(loader):

            # Discriminator part
            # input
            real = real.to(device)
            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            fake = gen(noise)
            # output
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            # loss
            disc_loss_real = loss(disc_real, torch.ones_like(disc_real))
            disc_loss_fake = loss(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = disc_loss_real + disc_loss_fake
            # Optimize
            disc_opt.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Genetator part
            generated_score = disc(fake)
            gen_loss = loss(generated_score, torch.ones_like(generated_score))
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} Batch {batch_index}/{len(loader)} : generator loss = {gen_loss}, discriminator loss = {disc_loss}")

                with torch.inference_mode():
                    test_image = gen(test_noise)
                    test_grid = vi.utils.make_grid(
                        test_image[:32], normalize=True)
                    real_grid = vi.utils.make_grid(real[:32], normalize=True)
                    real_writter.add_image(
                        "dataset image", real_grid, global_step=step
                    )
                    fake_writter.add_image(
                        "generated image", test_grid, global_step=step
                    )
                step += 1
