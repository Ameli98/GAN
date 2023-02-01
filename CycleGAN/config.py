import torch
import albumentations as alb

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
lambda_cycle = 10
lambda_identity = 0
batch_size = 1
num_workers = 4
epochs = 10

transform = alb.Compose(
    [
        alb.Resize(width=256, height=256),
        alb.HorizontalFlip(0.5),
        alb.Normalize(mean=[0.5, 0.5, 0.5], std=[
                      0.5, 0.5, 0.5], max_pixel_value=255),
        alb.pytorch.ToTensorV2(),
    ]
)
