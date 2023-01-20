import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import pathlib
import config
import numpy as np


class pix2pixDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = pathlib.Path(root_dir)
        self.image_list = list(self.root_dir.iterdir())

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # Open  the image
        image_path = self.root_dir / self.image_list[index]
        image = np.array(Image.open(image_path))
        # Split the image
        colored_image = image[:, :600, :]
        uncolored_image = image[:, 600:, :]

        # Argumentation part
        augmentations = config.both_transform(
            image=colored_image, image0=uncolored_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_label(image=target_image)["image"]

        return input_image, target_image


# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     from torchvision.utils import save_image, make_grid
#     dataset = pix2pixDataset("/mnt/c/Dataset/anime_coloring/train")
#     loader = DataLoader(dataset, 64)
#     for train_images, label_images in loader:
#         train_images = make_grid(train_images)
#         label_images = make_grid(label_images)
#         save_image(train_images, "train_images.jpg")
#         save_image(label_images, "label_images.jpg")
