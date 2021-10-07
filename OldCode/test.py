import time
import argparse

import torchvision
import torch

from torch.utils.tensorboard import SummaryWriter

from MyModel import VideoModel
from dataloader import datasets, CPU_GPU
from transform import transforms
import utils


from torchvision.transforms._transforms_video import (
    NormalizeVideo
)

from pytorchvideo.transforms import (
    #Normalize,
    ShortSideScale
)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test the GRASP network', usage='python3 test.py load_model_weights [options]', epilog='The code is not very well optimized :)')
    parser.add_argument('--labels_file', type=str, help='CSV file with labels', default='./own_grasp_labels.csv')
    parser.add_argument('--video_path', type=str, help='Path to the folder with the videos', default='/home/victor.sendinog/data-local/Videos-GoPro/')
    parser.add_argument('--batch_size', type=int, help='Batch size for the model', default=32)
    parser.add_argument('--num_workers', type=int, help='Number of threads for reading the dataset', default=4)
    parser.add_argument('--pin_memory', action='store_true', help='Specify the argument to use Pytorch pin memory. By default this option is not used.')
    parser.add_argument('--model_weights', type=str, help='Path to an existing model weights file to use as initial weights', required=True)

    args = parser.parse_args()

    device = CPU_GPU.get_default_device()

    save_directory = args.model_weights.rsplit('/', 1)[0] + "/test/"
    elems = args.model_weights.rsplit('/', 1)[-1].split('-')
    args.video_model = elems[0]
    args.categories = elems[1]

    print("-------------------------------")
    print("Configuration Settings:")
    print("-------------------------------")
    print("Device: ", device)
    print("Video model: ", args.video_model)
    print("Labels file: ", args.labels_file)
    print("Video path: ", args.video_path)
    print("Batch size: ", args.batch_size)
    print("Num workers: ", args.num_workers)
    print("Pin memory: ", args.pin_memory)
    print("Save directory: ", save_directory)
    print("Model weights: ", args.model_weights)
    print("Categories: ", args.categories)
    print("-------------------------------")

    num_categories = utils.get_num_categories(args.categories)

    model = VideoModel(args.video_model, num_categories, "no_convolution", "full").to(device=device)
    #if torch.cuda.device_count() > 1:
    #    print("Usage of", torch.cuda.device_count(), "GPUs")
    #    model = nn.DataParallel(model)

    dataset = datasets.GRASPDataset(
        csv_file = args.labels_file, 
        video_path = args.video_path,
        sampling_mode = "uniform",
        transform=torchvision.transforms.Compose([
            transforms.CSVToVideos(padding_mode="last"),
            NormalizeVideo(model.transform_params["mean"], model.transform_params["std"]),
            ShortSideScale(model.transform_params["crop_size"])
        ]),
        num_frames = model.transform_params["num_frames"],
        categories = args.categories
    )

    test_writer = SummaryWriter(save_directory)
    
    if(args.model_weights != ""):
        model.load_state_dict(torch.load(args.model_weights))

    print("-------------------------------")
    print("Dataset Information:")
    print("-------------------------------")  
    dataset_size = len(dataset)
    print('Dataset size: ', dataset_size)
    print("-------------------------------") 
        

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory= args.pin_memory, num_workers=args.num_workers)    
    

    #Create Grid of images to show the transformation
    dataiter = iter(test_loader)
    videos, labels = dataiter.next()
    vid1 = videos[0]
    vid2 = videos[16]
    vid3 = videos[24]
    vid1 = vid1.permute(1,0,2,3)
    vid2 = vid2.permute(1,0,2,3)
    vid3 = vid3.permute(1,0,2,3)
    img_grid1 = torchvision.utils.make_grid(vid1)
    img_grid2 = torchvision.utils.make_grid(vid2)
    img_grid3 = torchvision.utils.make_grid(vid3)
    test_writer.add_image('Test im 1', img_grid1)
    test_writer.add_image('Test im 2', img_grid2)
    test_writer.add_image('Test im 3', img_grid3)


    test_loader = CPU_GPU.DeviceDataLoader(test_loader, device)

    print("-------------------------------") 
    print("Test starts")
    print("-------------------------------") 
    model.eval()
    test_start_time = time.time()
    correct, total = 0, 0
    conf_mat = torch.zeros(num_categories, num_categories)
    with torch.no_grad():
        for i, data in enumerate(test_loader,0):
            inputs, targets = data

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            for t, p in zip(targets.view(-1), predicted.view(-1)):
                conf_mat[t.long(), p.long()] += 1

    test_end_time = time.time()
    print("Confusion matrix: ")
    print(conf_mat)
    fig = utils.plot_confusion_matrix(conf_mat, dataset.get_label_mapping())
    test_writer.add_figure('CM test', fig)
        
    print("Per class accuracy: ")
    print(conf_mat.diag()/conf_mat.sum(1))

    test_accuracy = 100*correct / total
    print('Test accuracy: ',test_accuracy,'%')
    test_writer.add_scalar('Test accuracy', test_accuracy, 0)
        
    print("*** Test time: ",test_end_time-test_start_time," ***")
    test_writer.add_scalar('Test time', test_end_time-test_start_time, 0)
    print("-------------------------------")
    
test_writer.close()
