import torch
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset


# Custom dataset class to load images without labels
class ImageDataset(Dataset):
    def __init__(self, image_directory, transform=None):
        self.image_directory = image_directory
        self.image_filenames = sorted(os.listdir(image_directory))
        self.transform = transform

    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_directory, image_filename)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_filenames)


def collate_fn(batch):
    return torch.stack(batch)
