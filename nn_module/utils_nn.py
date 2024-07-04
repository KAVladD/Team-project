import os
from random import choices

from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch, torchvision
from PIL import Image
import matplotlib.pyplot as plt


class RandomNDataset(Dataset):
    def __init__(self, data_path, transform, n_images=2000):
        self.path = data_path
        self.transform = transform
        self.data_size = n_images

        images_pathes = []
        data_folder = os.listdir(self.path)
        for image_name in data_folder:
            image_path = os.path.join(self.path, image_name)
            # image = cv2.imread(image_path)
            images_pathes.append(image_path)

        self.images_pathes = images_pathes
        self.new_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, val = self.data[index]
        return image, val

    def new_data(self):
        self.data = []
        random_pathes = choices(self.images_pathes, k=self.data_size)
        for path in random_pathes:
            image = self.transform(Image.open(path))
            val = torch.tensor(int(path.split("_")[5].split(".")[1]) / 10000)
            self.data.append((image, val))


class AllDataDataset(Dataset):
    def __init__(self, data_path, transform, n_images=2000):
        self.path = data_path
        self.transform = transform
        self.data_size = n_images

        images_pathes = []
        data_folder = os.listdir(self.path)
        for image_name in data_folder:
            image_path = os.path.join(self.path, image_name)
            # image = cv2.imread(image_path)
            images_pathes.append(image_path)

        self.images_pathes = images_pathes
        self.new_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, val = self.data[index]
        return image, val

    def new_data(self):
        self.data = []
        for path in self.images_pathes:
            image = self.transform(Image.open(path))
            val = torch.tensor(int(path.split("_")[5].split(".")[1]) / 10000)
            self.data.append((image, val))


def init_train_transform():

    train_transform = torchvision.transforms.Compose(
        [
            # Converting images to the size that the model expects
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),  # A RandomHorizontalFlip to augment our data
            torchvision.transforms.ToTensor(),  # Converting to tensor
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalizing the data to the data that the ResNet18 was trained on
        ]
    )

    return train_transform

def init_gray_train_transform():

    train_transform = torchvision.transforms.Compose(
        [
            # Converting images to the size that the model expects
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),  # A RandomHorizontalFlip to augment our data
            torchvision.transforms.ToTensor(),  # Converting to tensor
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalizing the data to the data that the ResNet18 was trained on
            torchvision.transforms.Grayscale(),
        ]
    )

    return train_transform


def init_val_transform():

    val_transform = torchvision.transforms.Compose(
        [
            # Converting images to the size that the model expects
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),  # Converting to tensor
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalizing the data to the data that the ResNet18 was trained on
        ]
    )

    return val_transform

def init_gray_val_transform():

    val_transform = torchvision.transforms.Compose(
        [
            # Converting images to the size that the model expects
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),  # Converting to tensor
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalizing the data to the data that the ResNet18 was trained on
            torchvision.transforms.Grayscale(),
        ]
    )

    return val_transform
