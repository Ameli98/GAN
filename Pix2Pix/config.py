import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = Path("/mnt/c/Dataset/anime_coloring/train")
val_dir = Path("/mnt/c/Dataset/anime_coloring/val")
val_imageset_dir = Path("./colored_image")
val_sample_dir = Path("./sample_image")
logfile_name = Path("logfile.log")
lr = 2e-4
batch_size = 64
num_workers = 4
image_size = 256
image_channel = 3
l1_lambda = 100
epochs = 500
model_checkpoint = Path("model.pth.tar")
load_model = True if model_checkpoint.exists() else False

both_transform = A.Compose(
    [A.Resize(width=256, height=256), A.HorizontalFlip(p=0.5),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                    0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_label = A.Compose(
    [
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                    0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
