import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-5
batch_size = 1
num_workers = 4
epochs = 30

lambda_cycle = 10
lambda_identity = 0

# For colab
dataset_path1 = Path("/content/Academic_Art")
dataset_path2 = Path("/content/cosplay")

# dataset_path1 = Path("/mnt/c/dataset/WikiArt_ArtMovements/Academic_Art")
# dataset_path2 = Path("/mnt/c/dataset/cosplay")

sample_root = Path("/content/Sample")
# sample_root = Path("/mnt/c/Sample/CycleGAN")
sample_gen1 = sample_root / "gen1"
sample_gen2 = sample_root / "gen2"
sample_cycle1 = sample_root / "cycle1"
sample_cycle2 = sample_root / "cycle2"

model_checkpoint = Path("model.pth.tar")
load_model = True if model_checkpoint.exists() else False

transform = alb.Compose(
    [
        alb.Resize(width=256, height=256),
        alb.HorizontalFlip(0.5),
        alb.Normalize(mean=[0.5, 0.5, 0.5], std=[
                      0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"}
)
