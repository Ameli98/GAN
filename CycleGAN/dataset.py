from torch import nn
from pathlib import Path
from PIL import Image
import numpy as np


class cycleGAN_dataset(nn.Module):
    def __init__(self, root1, root2, transform):
        super().__init__()
        self.image_root1 = Path(root1)
        self.image_root2 = Path(root2)
        self.image_list1 = list(self.image_root1.iterdir())
        self.image_list2 = list(self.image_root2.iterdir())
        self.transform = transform

        self.image_list1 = self.image_list1[:min(
            len(self.image_list1), len(self.image_list2))]
        self.image_list2 = self.image_list2[:min(
            len(self.image_list1), len(self.image_list2))]

    def __len__(self):
        return len(self.image_list1)

    def __getitem__(self, index):
        image_path1 = self.image_root1 / \
            self.image_list1[index % len(self.image_list1)]
        image_path2 = self.image_root2 / \
            self.image_list2[index % len(self.image_list2)]
        original_image1 = np.array(Image.open(image_path1).convert("RGB"))
        original_image2 = np.array(Image.open(image_path2).convert("RGB"))
        albumentations = self.transform(
            image=original_image1, image0=original_image2)
        image1 = albumentations["image"]
        image2 = albumentations["image0"]
        return image1, image2
