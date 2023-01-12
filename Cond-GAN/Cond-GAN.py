import torch
from Model import *
import torchvision as vi
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs, batch_size, lr = 5, 64, 1e-4
    image_size, image_channels,  = 64, 1
    noise_dim, features_d, features_g = 100, 64, 64
    lambda_gp, critic_iteration = 10, 5
    num_classes, emd_size = 10, 100
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

    # Generator & Critic
    critic = Critic(image_channels, features_d,
                    image_size, num_classes).to(device)
    gen = Generator(noise_dim, num_classes, emd_size,
                    image_channels, features_g).to(device)

    # Dataset & Dataloader
    # dataset = vi.datasets.ImageFolder(
    #     root="./images",
    #     transform=transforms,
    # )
    dataset = vi.datasets.MNIST(
        root="datasets", transform=transforms, download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Optimizer
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0, 0.9))
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0, 0.9))

    # SummaryWritter
    real_writter = SummaryWriter(f"log/real")
    fake_writter = SummaryWriter(f"log/fake")
    step = 0

    # Test run
    # real, label = next(iter(loader))

    # Trainning loop
    for epoch in range(epochs):
        for batch_index, (real, label) in enumerate(loader):

            # critic part
            real = real.to(device)
            label = label.to(device)
            real_size = real.shape[0]
            for i in range(critic_iteration):
                # input
                noise = torch.randn(real_size, noise_dim, 1, 1).to(device)
                fake = gen(noise, label)
                gp = gradient_penalty(critic, label, real, fake, device)
                # output
                critic_real = critic(real, label).reshape(-1)
                critic_fake = critic(fake, label).reshape(-1)
                # loss
                critic_loss = -(torch.mean(critic_real) -
                                torch.mean(critic_fake) - gp*lambda_gp)
                # Optimize
                critic.zero_grad()
                critic_loss.backward(retain_graph=True)
                critic_opt.step()

            # Genetator part
            generated_score = critic(fake, label).reshape(-1)
            gen_loss = -torch.mean(generated_score)
            gen.zero_grad()
            gen_loss.backward()
            gen_opt.step()
            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} Batch {batch_index}/{len(loader)} : generator loss = {gen_loss}, critic loss = {critic_loss}")
                with torch.inference_mode():
                    test_image = gen(noise, label)
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
