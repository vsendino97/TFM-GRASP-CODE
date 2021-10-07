import torchvision
from base import BaseDataLoader
from .datasets import VideoDataset, TransformedVideoDataset, TransformedImageDataset, ImageDataset, HandDataset
from transform import transforms


from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)

from torchvision.transforms import (
    Normalize,
    Resize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation
)
from pytorchvideo.transforms import (
    # Normalize,
    ShortSideScale,
    UniformCropVideo,
    RandomShortSideScale,
    ShortSideScale
)

class VideoDataLoader(BaseDataLoader):
    """
    Video data loading from the original video
    """
    def __init__(self, csv_dir, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, num_frames=8, categories="BigCategories"):

        data_transforms = {
            'train': torchvision.transforms.Compose([
                transforms.CSVToVideos(padding_mode="last", sampling_mode="random", num_frames=num_frames),
                NormalizeVideo((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                ShortSideScale(320),
                RandomCrop(244),
                RandomRotation(degrees=(-90, 90)),
                RandomHorizontalFlip(p=0.5)

            ]),
            'val': torchvision.transforms.Compose([
                transforms.CSVToVideos(padding_mode="last", sampling_mode="uniform", num_frames=num_frames),
                NormalizeVideo((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                ShortSideScale(320),
                CenterCropVideo(244)
           ]),
        }

        self.data_dir = data_dir
        
        datasets = {x: VideoDataset(csv_file=csv_dir, video_path=data_dir,
                                             num_frames=num_frames, categories=categories, transform=data_transforms[x])  for x in ['train', 'val']}

        super().__init__(datasets, batch_size, shuffle, validation_split, num_workers)


class HybridDataLoader(BaseDataLoader):
    """
    Video data loading from the original video
    """
    def __init__(self, csv_dir, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, num_frames=8, categories="BigCategories"):

        data_transforms = {
            'train': torchvision.transforms.Compose([
                transforms.CSVToVideos(padding_mode="last", sampling_mode="random", num_frames=num_frames),

            ]),
            'val': torchvision.transforms.Compose([
                transforms.CSVToVideos(padding_mode="last", sampling_mode="uniform", num_frames=num_frames),
           ]),
        }

        self.data_dir = data_dir

        datasets = {x: VideoDataset(csv_file=csv_dir, video_path=data_dir,
                                             num_frames=num_frames, categories=categories, transform=data_transforms[x])  for x in ['train', 'val']}

        super().__init__(datasets, batch_size, shuffle, validation_split, num_workers)



class TransformedDataLoader(BaseDataLoader):
    """
    Video data loading from the transformed sources.
    """
    def __init__(self, csv_dir, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, num_frames=8, categories="BigCategories", sampling_mode="uniform"):

        transform=torchvision.transforms.Compose([
            transforms.ReadTransformedVideos(num_frames),
            RandomShortSideScale(min_size=256, max_size=320),
            RandomCrop(244)
        ])

        self.data_dir = data_dir
        self.dataset = TransformedVideoDataset(csv_file=csv_dir, video_path=data_dir, sampling_mode=sampling_mode,
                                             num_frames=num_frames, categories=categories, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class ImageDataLoader(BaseDataLoader):
    """
    Video data loading from the transformed sources.
    """
    def __init__(self, csv_dir, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, num_frames=8, categories="BigCategories", sampling_mode="uniform"):

        transform = torchvision.transforms.Compose([
            transforms.CSVToImages(),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            Resize(320),
            RandomCrop(244)
        ])

        self.data_dir = data_dir
        #self.dataset = TransformedImageDataset(csv_file=csv_dir, video_path=data_dir, sampling_mode=sampling_mode,
        #                                     num_frames=num_frames, categories=categories, transform=transform)
        self.dataset = ImageDataset(csv_file=csv_dir, video_path=data_dir, sampling_mode=sampling_mode,
                                               num_frames=num_frames, categories=categories, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class HandDataLoader(BaseDataLoader):
    """
    Video data loading from the transformed sources.
    """
    def __init__(self, csv_dir, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, num_frames=8, categories="BigCategories", sampling_mode="uniform"):

        data_transforms = {
            'train': torchvision.transforms.Compose([
                transforms.ReadDiskImage(),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                Resize([244,244]),
                RandomRotation(degrees=(-90, 90)),
                RandomHorizontalFlip(p=0.5)

            ]),
            'val': torchvision.transforms.Compose([
                transforms.ReadDiskImage(),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                Resize([244,244]),
           ]),
        }

        self.data_dir = data_dir
        datasets = {x: HandDataset(csv_file=csv_dir, video_path=data_dir,
                                             num_frames=1, categories=categories, transform=data_transforms[x])  for x in ['train', 'val']}

        super().__init__(datasets, batch_size, shuffle, validation_split, num_workers)

