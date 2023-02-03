import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import config
from generator import generator
from discriminator import discriminator
from dataset import cycleGAN_dataset
import tqdm

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    gen1 = generator().to(config.device)
    gen2 = generator().to(config.device)
    gen_opt = disc_opt = torch.optim.Adam(
        list(gen1.parameters()) + list(gen2.parameters()), lr=config.lr, betas=(0.5, 0.999))

    disc1 = discriminator().to(config.device)
    disc2 = discriminator().to(config.device)
    disc_opt = torch.optim.Adam(
        list(disc1.parameters()) + list(disc2.parameters()), lr=config.lr, betas=(0.5, 0.999))

    mse = nn.MSELoss()
    l1loss = nn.L1Loss()

    dataset = cycleGAN_dataset(
        config.dataset_path1, config.dataset_path2, config.transform)
    dataloader = DataLoader(dataset, config.batch_size, shuffle=True)

    # loading checkpoint
    if config.load_model:
        checkpoint = torch.load(config.model_checkpoint)
        disc1.load_state_dict(checkpoint["disc1_model"])
        disc2.load_state_dict(checkpoint["disc2_model"])
        disc_opt.load_state_dict(checkpoint["disc_opt"])
        for param_group in disc_opt.param_groups:
            param_group["lr"] = config.lr
        gen1.load_state_dict(checkpoint["gen1_model"])
        gen2.load_state_dict(checkpoint["gen2_model"])
        gen_opt.load_state_dict(checkpoint["gen_opt"])
        for param_group in gen_opt.param_groups:
            param_group["lr"] = config.lr

    if not config.sample_root.exists():
        config.sample_root.mkdir()
        config.sample_gen1.mkdir()
        config.sample_gen2.mkdir()
        config.sample_cycle1.mkdir()
        config.sample_cycle2.mkdir()

    # Training loop
    for epoch in range(config.epochs):
        loop = tqdm.tqdm(dataloader, ncols=100)
        for index, (image1, image2) in enumerate(loop):
            loop.set_description(f"Epoch:{epoch+1}/{config.epochs}")
            image1 = image1.to("cuda")
            image2 = image2.to("cuda")

            # discriminator part
            gen_image1 = gen1(image2)
            gen_image2 = gen2(image1)

            # adversial loss
            disc_gen1 = disc1(gen_image1)
            disc_gen2 = disc2(gen_image2)
            disc_real1 = disc1(image1)
            disc_real2 = disc2(image2)

            disc_real_loss1 = mse(disc_real1, torch.ones_like(disc_real1))
            disc_fake_loss1 = mse(disc_gen1, torch.zeros_like(disc_gen1))
            disc_real_loss2 = mse(disc_real2, torch.ones_like(disc_real2))
            disc_fake_loss2 = mse(disc_gen2, torch.zeros_like(disc_gen2))
            disc_loss = (disc_real_loss1 + disc_fake_loss1 +
                         disc_real_loss2 + disc_fake_loss2) / 2

            disc_opt.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # generator part
            # adversial loss
            disc_gen1 = disc1(gen_image1)
            disc_gen2 = disc2(gen_image2)
            gen_loss1 = mse(disc_gen1, torch.zeros_like(disc_gen1))
            gen_loss2 = mse(disc_gen2, torch.zeros_like(disc_gen2))

            # cycle loss
            cycle_image1 = gen1(gen_image2)
            cycle_image2 = gen2(gen_image1)
            cycle_loss1 = l1loss(image1, cycle_image1)
            cycle_loss2 = l1loss(image2, cycle_image2)
            cycle_loss = cycle_loss1 + cycle_loss2

            gen_loss = gen_loss1 + gen_loss2 + config.lambda_cycle * cycle_loss
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            # Show the loss
            loop.set_postfix(disc_loss=disc_loss.item(),
                             gen_loss=gen_loss.item())

            # Save generated images
            gen_image1 = gen_image1 * 0.5 + 0.5
            gen_image2 = gen_image2 * 0.5 + 0.5
            cycle_image1 = cycle_image1 * 0.5 + 0.5
            cycle_image2 = cycle_image2 * 0.5 + 0.5
            save_image(gen_image1, config.sample_gen1 / f"{index}.jpg")
            save_image(gen_image2, config.sample_gen2 / f"{index}.jpg")
            save_image(cycle_image1, config.sample_cycle1 / f"{index}.jpg")
            save_image(cycle_image2, config.sample_cycle2 / f"{index}.jpg")

        # Checkpoint of models
        model_state = {
            "disc1_model": disc1.state_dict(),
            "disc2_model": disc2.state_dict(),
            "disc_opt": disc_opt.state_dict(),
            "gen1_model": gen1.state_dict(),
            "gen2_model": gen2.state_dict(),
            "gen_opt": gen_opt.state_dict(),
        }
        torch.save(model_state, config.model_checkpoint)
