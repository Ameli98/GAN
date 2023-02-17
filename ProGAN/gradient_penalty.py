import torch


def gradient_penalty(discriminator, real, fake, phase, alpha, device="cpu") -> float:
    batch, C, H, W = real.shape
    weight = torch.rand(batch, 1, 1, 1).repeat(1, C, H, W).to(device)
    interpolated_image = real * weight + fake * (1 - weight)
    mix_score = discriminator(interpolated_image, phase, alpha)
    gradient = torch.autograd.grad(
        outputs=mix_score, inputs=interpolated_image,
        grad_outputs=torch.ones_like(mix_score),
        retain_graph=True, create_graph=True)[0]
    grad_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((grad_norm - 1) ** 2)
    return penalty
