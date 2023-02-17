from pathlib import Path
import torch
from torchvision.transforms import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-3
batch_size = 4
num_workers = 4
epochs = 30
lambda_gp = 10

dataset_path = Path("/mnt/c/dataset/gfskin")
sample_root = Path("/mnt/c/Sample/ProGAN/gfskin")


model_checkpoint = Path("model.pth.tar")
load_model = True if model_checkpoint.exists() else False

# notification_sound = Path("/mnt/d/SoundEffect/notification.mp3")


def transform(img_size):
    transformation = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
                0.5, 0.5, 0.5]),
        ]
    )
    return transformation
