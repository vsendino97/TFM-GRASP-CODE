import argparse

import torchvision
import torch

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataloader import datasets, CPU_GPU
from transform import transforms

from torchvision.transforms._transforms_video import (
    NormalizeVideo
)

from pytorchvideo.transforms import (
    #Normalize,
    ShortSideScale
)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the GRASP network', usage='python3 train.py [options]', epilog='The code is not very well optimized :)')
    parser.add_argument('--num_frames', type=int, help='Specify the number of images to save for each grasp', default = 8)
    parser.add_argument('--labels_file', type=str, help='CSV file with labels', default='data/nature_dataset/nature_final_labels.csv')
    parser.add_argument('--video_path', type=str, help='Path to the folder with the videos', default='/home/victor.sendinog/data-local/Gopro/')
    parser.add_argument('--save_directory', type=str, help='Path used for saving the images', default='/home/victor.sendinog/data-local/nature-transformations/')
    parser.add_argument('--num_aug', type=int, help='Number of augmentations performed per each image', default=1)
    parser.add_argument('--save_csv', type=str, help='Path to save the csv with augmented labels', default='data/nature_dataset/augmented_dataset.csv')
    #--labels_file /home/victor.sendinog/data-local/Yale-Grasp-Dataset/yale_final_labels.csv --video_path /home/victor.sendinog/data-local/Yale-Grasp-Dataset/ --save_directory /home/victor.sendinog/data-local/yale-transformations/

    args = parser.parse_args()

    device = CPU_GPU.get_default_device()

    print("-------------------------------")
    print("Configuration Settings:")
    print("-------------------------------")
    print("Device: ", device)
    print("Num frames: ", args.num_frames)
    print("Labels file: ", args.labels_file)
    print("Video path: ", args.video_path)
    print("Save directory: ", args.save_directory)
    print("Num augmentations: ", args.num_aug)
    print("Save csv: ", args.save_csv)
    print("-------------------------------")


    dataset = datasets.GenerateTransformedDataset(
        read_csv = args.labels_file,
        save_csv = args.save_csv,
        video_path = args.video_path,
        transform=torchvision.transforms.Compose([
            transforms.CSVToVideos(padding_mode="last"),
            NormalizeVideo((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            ShortSideScale(320),
            #RandomShortSideScale(min_size=256, max_size=320),
            #RandomCrop(244)
            #RandomHorizontalFlip(p=0.5),
        ]),

        num_frames = args.num_frames,
        num_aug = args.num_aug
    )


    print("-------------------------------")
    print("Dataset Information:")
    print("-------------------------------")  
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    print('Dataset size: ', dataset_size)

    train_loader = torch.utils.data.DataLoader(dataset)
    print("-------------------------------") 
    print("Transformation starts")
    print("-------------------------------")
    #train_writer = SummaryWriter("./transformed_seq/")
    for i, data in enumerate(train_loader, 0):
        inputs = data
        s_dir = args.save_directory + '{}' + '.pt'
        torch.save(inputs, s_dir.format(i))
        print(s_dir.format(i))
        
        #vid1 = torch.squeeze(inputs)
        #print(vid1.size())
        #vid1 = vid1.permute(1,0,2,3)
        #img_grid1 = torchvision.utils.make_grid(vid1)
        #train_writer.add_image('im'+ str(i), img_grid1) 
