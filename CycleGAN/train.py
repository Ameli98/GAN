import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import config
from generator import generator
from discriminator import discriminator
from dataset import cycleGAN_dataset
import tqdm

if __name__ == "__main__":
    gen1 = generator()
    gen2 = generator()
    gen_opt = disc_opt = torch.optim.Adam(
        list(gen1.parameters()) + list(gen2.parameters(), lr=config.lr, betas=(0.5, 0.999)))

    disc1 = discriminator()
    disc2 = discriminator()
    disc_opt = torch.optim.Adam(
        list(disc1.parameters()) + list(disc2.parameters()), lr=config.lr, betas=(0.5, 0.999))

    mse = nn.MSELoss()
    l1loss = nn.L1Loss()

    dataset = cycleGAN_dataset(
        config.dataset_path1, config.dataset_path2, config.transform)
    dataloader = DataLoader(dataset, config.batch_size, shuffle=True)
    loop = tqdm.tqdm(dataloader)

    for epoch in range(config.epochs):
        for index, (image1, image2) in enumerate(loop):
            gen_image1 = gen1(image2)
            gen_image2 = gen2(image1)

            disc_gen1 = disc1(gen_image1)
            disc_gen2 = disc2(gen_image2)
            disc_real1 = disc1(image1)
            disc_real2 = disc2(image2)
            disc_loss1 = mse(disc_real1, torch.ones_like(
                disc_real1)) + mse(disc_gen1, torch.zeros_like(disc_gen1))
            disc_loss2 = mse(disc_real2, torch.ones_like(
                disc_real2)) + mse(disc_gen2, torch.zeros_like(disc_gen2))
            disc_loss = (disc_loss1 + disc_loss2) / 2

            cycle_image1 = gen1(gen_image2)
            cycle_image2 = gen2(gen_image1)
            cycle_loss = l1loss(image1, cycle_image1) + \
                l1loss(image2, cycle_image2)

            disc_loss = disc_loss + cycle_loss
            disc1.zero_grad()
            disc2.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            disc_gen1 = disc1(gen_image1)
            disc_gen2 = disc2(gen_image2)
            gen_loss1 = mse(disc_gen1, torch.zeros_like(disc_gen1))
            gen_loss2 = mse(disc_gen2, torch.zeros_like(disc_gen2))
            gen_loss = gen_loss1 + gen_loss2
            gen1.zero_grad()
            gen2.zero_grad()
            gen_loss.backward()
            gen_opt.step()
