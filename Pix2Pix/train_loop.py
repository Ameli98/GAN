import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pathlib
from tqdm import tqdm
import config
from Discriminator import Discriminator
from Generator import Generator
from dataset import pix2pixDataset

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    disc = Discriminator()
    disc_opt = torch.optim.Adam(
        disc.parameters(), lr=config.lr, betas=(0.5, 0.999))
    gen = Generator()
    gen_opt = torch.optim.Adam(
        gen.parameters(), lr=config.lr, betas=(0.5, 0.999))
    bceloss_func = nn.BCEWithLogitsLoss()
    train_dataset = pix2pixDataset(config.train_dir)
    train_loader = DataLoader(
        train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataset = pix2pixDataset(config.val_dir)
    val_loader = DataLoader(val_dataset, config.batch_size,
                            num_workers=config.num_workers)
