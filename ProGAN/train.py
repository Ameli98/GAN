import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from generator import Generator
from discriminator import Discriminator
from gradient_penalty import gradient_penalty as gp
import config as cg
import tqdm
from sys import argv

torch.backends.cudnn.benchmarks = True

if __name__ == "__main__":
    # initialization
    g = Generator().to(cg.device)
    g_opt = Adam(g.parameters(), lr=cg.lr, betas=(0, 0.99))
    d = Discriminator().to(cg.device)
    d_opt = Adam(d.parameters(), lr=cg.lr, betas=(0, 0.99))

    phase = int(argv[1]) if len(argv) >= 2 else 0
    # loading model
    if cg.load_model:
        cpt = torch.load(cg.model_checkpoint)
        g.load_state_dict(cpt["g"])
        g_opt.load_state_dict(cpt["g_opt"])
        for param_group in g_opt.param_groups:
            param_group["lr"] = cg.lr
        d.load_state_dict(cpt["d"])
        d_opt.load_state_dict(cpt["d_opt"])
        for param_group in d_opt.param_groups:
            param_group["lr"] = cg.lr

    # mkdir sample directory
    cg.sample_root.mkdir(parents=True, exist_ok=True)

    # training loop
    for step in range(phase, 7):
        alpha = 0
        dataset = ImageFolder(cg.dataset_path, cg.transform(2**(step+2)))
        loader = DataLoader(dataset, cg.batch_size,
                            shuffle=True, num_workers=cg.num_workers)
        for epoch in range(cg.epochs):
            alpha = min(alpha, 1)
            loop = tqdm.tqdm(loader)
            loop.set_description(f"Step:{step}/6 Epoch:{epoch+1}/{cg.epochs}")
            for index, (real, label) in enumerate(loop):
                # generate image
                real = real.to(cg.device)
                noise = torch.randn(real.shape[0], 512, 1, 1).to(cg.device)
                fake = g(noise, step, alpha)

                # discriminator part
                d_real = d(real, step, alpha)
                d_fake = d(fake, step, alpha)
                d_loss = torch.mean(d_fake) - torch.mean(d_real) + \
                    cg.lambda_gp * gp(d, real, fake, step, alpha, cg.device)

                d_opt.zero_grad()
                d_loss.backward(retain_graph=True)
                d_opt.step()

                # generator part
                d_fake = d(fake, step, alpha)
                g_loss = - torch.mean(d_fake)

                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

                # show the loss
                loop.set_postfix(disc_loss=d_loss.item(),
                                 gen_loss=g_loss.item())

                # save model
                model_state = {"g": g.state_dict(),
                               "g_opt": g_opt.state_dict(),
                               "d": d.state_dict(),
                               "d_opt": d_opt.state_dict(),
                               }
                torch.save(model_state, cg.model_checkpoint)
                # save image
                gen_image = fake * 0.5 + 0.5
                save_image(gen_image, cg.sample_root / f"{index}.jpg")

            # update alpha
            alpha += 4/cg.epochs
