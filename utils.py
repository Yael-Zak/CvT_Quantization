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


def get_deepest_folder(directory):
    subdirectories = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    if not subdirectories:
        return directory
    else:
        deepest_subdirectory = max(subdirectories, key=lambda f: get_deepest_folder(os.path.join(directory, f)).count(os.path.sep))
        return get_deepest_folder(os.path.join(directory, deepest_subdirectory))


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels
