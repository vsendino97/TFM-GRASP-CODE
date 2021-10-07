import time
import argparse
import os, psutil
import torchvision
import torch
import torch.optim as optim

import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

import torchvideo

from dataloader import datasets, CPU_GPU
from transform import transforms
import utils

from torchvision.utils import make_grid



from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)

from torchvision.transforms import (
    Normalize,
    Resize,
    RandomCrop,
    RandomHorizontalFlip
)
from pytorchvideo.transforms import (
    # Normalize,
    ShortSideScale,
    UniformCropVideo,
    RandomShortSideScale,
    ShortSideScale
)

if __name__ == '__main__':

    writer = SummaryWriter("./aux/")


    dataset = datasets.VideoDataset(
        csv_file="../data/yale_dataset/yale_final_labels.csv",
        video_path="data/yale_dataset/YaleDataset/",
        sampling_mode="uniform",
        transform=torchvision.transforms.Compose([
            transforms.CSVToVideos(padding_mode="last"),
            #NormalizeVideo((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            ShortSideScale(320),
            RandomShortSideScale(min_size=256, max_size=320),
            #RandomCrop(244)
        ]),
        num_frames=8,
        categories="SmallCategories",
    )

    validation_split = 0.2
    shuffle_dataset = False
    random_seed = 42

    """
    dataset_size = len(dataset)
    print('Dataset size: ', dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))


    train_indices, val_indices = indices[split:], indices[:split]
    print('Train size: ', len(train_indices))
    print(train_indices)
    print('Validation size: ', len(val_indices))
    print(val_indices)
    print("-------------------------------")

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    """

    #train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_sampler,num_workers=4)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if (batch_idx == 0):
                print("Enter")
                # Video input
                if (len(data.shape) == 5):
                    writer.add_video(str(np.random.randint(100)), data.cpu().permute(0, 2, 1, 3, 4))
                # Image input
                elif (len(data.shape) == 4):
                    writer.add_image('input', make_grid(data.cpu(), normalize=True))
            break

    writer.close()

