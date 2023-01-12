import torch
import torch.nn as nn
import torchvision as vi
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layer(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.04),
            nn.Linear(128, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    # initialization
    lr = 1e-3
    z_dim, img_dim = 64, 784
    batch_size, epochs = 28, 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    disc = Discriminator(img_dim).to(device)
    gen = Generator(z_dim, img_dim).to(device)
    transform = vi.transforms.Compose(
        [
            vi.transforms.ToTensor(),
            vi.transforms.Normalize((0.5,), (0.5,))
        ]
    )

    dataset = vi.datasets.MNIST(
        root="dataset/", download=True, transform=transform)
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)
    loss = nn.BCELoss()
    writer_fake = SummaryWriter(f"logs/fake")
    writer_real = SummaryWriter(f"logs/real")
    step = 0

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)

            # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_fake = disc(fake).view(-1)
            loss_fake = loss(disc_fake, torch.zeros_like(disc_fake))
            disc_real = disc(real).view(-1)
            loss_real = loss(disc_real, torch.ones_like(disc_real))
            loss_disc = (loss_fake + loss_real) / 2
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).view(-1)
            loss_gen = loss(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )
                # Testing
                with torch.inference_mode():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = vi.utils.make_grid(
                        fake, normalize=True)
                    img_grid_real = vi.utils.make_grid(
                        data, normalize=True)

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "Mnist Real Images", img_grid_real, global_step=step
                    )
                    step += 1
