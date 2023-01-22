import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
# import logging
import tqdm
import config
from Discriminator import Discriminator
from Generator import Generator
from dataset import pix2pixDataset

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    # Basic structure
    disc = Discriminator().to(device=config.device)
    disc_opt = torch.optim.Adam(
        disc.parameters(), lr=config.lr, betas=(0.5, 0.999))
    gen = Generator().to(device=config.device)
    gen_opt = torch.optim.Adam(
        gen.parameters(), lr=config.lr, betas=(0.5, 0.999))
    bceloss_func = nn.BCEWithLogitsLoss()
    l1loss_func = nn.L1Loss()
    train_dataset = pix2pixDataset(config.train_dir)
    train_loader = DataLoader(
        train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataset = pix2pixDataset(config.val_dir)
    val_loader = DataLoader(val_dataset, config.batch_size,
                            num_workers=config.num_workers)
    # Logging file
    # logging.basicConfig(filename=config.logfile_name,
    #                     format="%(asctime)s %(levelname)s %(message)s")
    # logfile = logging.getLogger('PIL').setLevel(logging.WARNING)

    # Loading previous model
    if config.load_model:
        checkpoint = torch.load(config.model_checkpoint)
        disc.load_state_dict(checkpoint["disc_model"])
        disc_opt.load_state_dict(checkpoint["disc_opt"])
        for param_group in disc_opt.param_groups:
            param_group["lr"] = config.lr
        gen.load_state_dict(checkpoint["gen_model"])
        gen_opt.load_state_dict(checkpoint["gen_opt"])
        for param_group in gen_opt.param_groups:
            param_group["lr"] = config.lr

    # Pre-process for recording sample generated image
    if not config.val_imageset_dir.exists():
        config.val_imageset_dir.mkdir()
    for index, (colored_image, uncolored_image) in enumerate(val_loader):
        original_image = make_grid(colored_image, nrow=4)
        save_image(original_image, config.val_imageset_dir /
                   f"set{index}.png")

    if not config.val_sample_dir.exists():
        config.val_sample_dir.mkdir()

    for epoch in range(config.epochs):
        # Training part
        # Check models' are in the training mode
        # if not (disc.training and gen.training):
        #     logfile.error("Models stuck in the eval mode")
        # loading bar
        loop = tqdm.tqdm(train_loader, leave=True)
        for index, (colored_image, uncolored_image) in enumerate(loop):
            colored_image = colored_image.to(config.device)
            uncolored_image = uncolored_image.to(config.device)
            # Discriminator part
            gen_image = gen(uncolored_image)
            disc_real = disc(uncolored_image, colored_image)
            disc_fake = disc(uncolored_image, gen_image)
            disc_loss = (bceloss_func(disc_real, torch.ones_like(
                disc_real)) + bceloss_func(disc_fake, torch.zeros_like(disc_fake))) / 2
            disc.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Generator part
            disc_real = disc(uncolored_image, colored_image)
            disc_fake = disc(uncolored_image, gen_image)
            bceloss = bceloss_func(disc_real, torch.ones_like(
                disc_real)) + bceloss_func(disc_fake, torch.zeros_like(disc_fake))
            gen_loss = bceloss + \
                config.l1_lambda * l1loss_func(gen_image, colored_image)
            gen.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            # Update loading bar
            loop.set_description(
                f"epoch: {epoch+1}/{config.epochs}")
            loop.set_postfix(disc_loss=disc_loss.item(),
                             gen_loss=gen_loss.item())

        # Validation part
        for index, (colored_image, uncolored_image) in enumerate(val_loader):
            colored_image = colored_image.to(config.device)
            uncolored_image = uncolored_image.to(config.device)
            with torch.inference_mode():
                # Record sample of generated image
                gen_image = gen(uncolored_image)
                gen_image = gen_image * 0.5 + 0.5
                gen_imageset = make_grid(gen_image, nrow=4)
                save_image(gen_image, config.val_sample_dir /
                           f"image{index}.png")
                # Checkpoint of models
                model_state = {
                    "disc_model": disc.state_dict(),
                    "disc_opt": disc_opt.state_dict(),
                    "gen_model": gen.state_dict(),
                    "gen_opt": gen_opt.state_dict(),
                }
                torch.save(model_state, config.model_checkpoint)
