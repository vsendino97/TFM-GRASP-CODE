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

from MyModel import VideoModel
from dataloader import datasets, CPU_GPU
from transform import transforms
import utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the GRASP network', usage='python3 train.py [options]', epilog='The code is not very well optimized :)')
    parser.add_argument('--video_model', type=str, help='Specify the model name: r2plus1d | slow_r50 | csn | x3d_m', default = 'slow_r50')
    parser.add_argument('--labels_file', type=str, help='CSV file with labels', default='./grasp-labels.csv')
    parser.add_argument('--video_path', type=str, help='Path to the folder with the videos', default='/home/victor.sendinog/data-local/Gopro/')
    parser.add_argument('--batch_size', type=int, help='Batch size for the model', default=32)
    parser.add_argument('--cross_val', action='store_true', help='Specify the argument for doing cross-validation. Otherwise holdout is performed')
    parser.add_argument('--num_folds', type=int, help='Number of cross-validation folds', default=5)
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs', default=5)
    parser.add_argument('--learn_rate', type=float, help='Learning rate applied in training', default=0.01)
    parser.add_argument('--num_workers', type=int, help='Number of threads for reading the dataset', default=4)
    parser.add_argument('--pin_memory', action='store_true', help='Specify the argument to use Pytorch pin memory. By default this option is not used.')
    parser.add_argument('--save_directory', type=str, help='Path used for saving the results', default='./results')
    parser.add_argument('--model_weights', type=str, help='Path to an existing model weights file to use as initial weights', default='')
    parser.add_argument('--num_categories', type=int, help='Number of category labels. Choose between 4 or 8', default=4)


    args = parser.parse_args()

    device = CPU_GPU.get_default_device()

    print("-------------------------------")
    print("Configuration Settings:")
    print("-------------------------------")
    print("Device: ", device)
    print("Video model: ", args.video_model)
    print("Labels file: ", args.labels_file)
    print("Video path: ", args.video_path)
    print("Batch size: ", args.batch_size)
    print("Cross val: ", args.cross_val)
    print("Num folds: ", args.num_folds)
    print("Num epochs: ", args.num_epochs)
    print("Learning rate: ", args.learn_rate)
    print("Num workers: ", args.num_workers)
    print("Pin memory: ", args.pin_memory)
    model_name = args.video_model + time.strftime("-%Y%m%d-%H%M%S-") + str(args.num_epochs) + "ep"
    save_directory = args.save_directory + "/" + args.video_model + "/" + model_name + "/"
    print("Save directory: ", save_directory)
    print("Model weights: ", args.model_weights)
    print("Number of categories: ", args.num_categories)
    print("-------------------------------")

    prog_start_time = time.time()

    model = VideoModel(args.video_model, args.num_categories).to(device=device)
    if torch.cuda.device_count() > 1:
        print("Usage of", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    dataset = datasets.VideoLabelGRASP(
        csv_file = args.labels_file, 
        video_path = args.video_path,
        sampling_mode = "uniform",
        transform=torchvision.transforms.Compose([
            transforms.CSVToVideos(num_frames=model.transform_params["num_frames"], padding_mode="last"),
            transforms.VideoResize([model.transform_params["side_size"], model.transform_params["side_size"]]),
            #transforms.VideoRandomCrop([112, 112]),
            torchvideo.transforms.NormalizeVideo(model.transform_params["mean"], model.transform_params["std"])
            ]),
        num_frames = model.transform_params["num_frames"],
        num_categories = args.num_categories
    )
    writer = SummaryWriter(save_directory)
    
    data_end_time = time.time()
    print("*** Dataset time ", data_end_time-prog_start_time, " ***")
    torch.save(model.state_dict(), 'original-weights.pth')
    #torch.cuda.empty_cache()
    #print(torch.cuda.memory_summary(device=device))

    if(args.model_weights != ""):
        model.load_state_dict(torch.load(args.model_weights))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learn_rate, momentum=0.9)

    #Cross-Validation Method
    if (args.cross_val):
        k_folds = args.num_folds
        results = {}
        torch.manual_seed(42)

        # Define K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)
        print("-------------------------------")     
        print("Cross-Validation start")
        print("-------------------------------")

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            print(f'FOLD {fold}')
            print("-------------------------------") 

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size, sampler=train_subsampler)
            val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size, sampler=val_subsampler)

            train_loader = CPU_GPU.DeviceDataLoader(train_loader, device)
            val_loader = CPU_GPU.DeviceDataLoader(val_loader, device)

            # Reset to the original weights
            model.load_state_dict(torch.load('original-weights.pth'))
            model.train()

            # Initialize optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Run the training loop for defined number of epochs
            for epoch in range(args.num_epochs):

                print(f'Starting epoch {epoch + 1}')
                running_loss = 0.0
                ep_start_time = time.time()
                # Iterate over the DataLoader for training data
                for i, data in enumerate(train_loader, 0):

                    # Get inputs
                    inputs, targets = data
                    # Zero the gradients
                    optimizer.zero_grad()
                    # Perform forward pass
                    outputs = model(inputs)
                    # Compute loss
                    loss = criterion(outputs, targets)

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    optimizer.step()

                    # Print statistics
                    running_loss += loss.item()
                    #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, current_loss))
                    """
                    if i % 500 == 499:
                        print('Loss after mini-batch %5d: %.3f' %
                              (i + 1, current_loss / 500))
                        current_loss = 0.0

                    """

                ep_end_time = time.time()
                writer.add_scalar(' Fold ' + str(fold+1) + ' epochs time', ep_end_time- ep_start_time, epoch+1)
                print(' Fold ' + str(fold+1) + ' epochs time', ep_end_time- ep_start_time, epoch+1)
                writer.add_scalar(' Fold ' + str(fold+1) + ' training loss', running_loss, epoch+1)
                print(' Fold ' + str(fold+1) + ' training loss', running_loss, epoch+1)


            # Process is complete. Save the model
            print('Training CV epoch has finished. Saving trained model.')
            
            save_path = save_directory + f'model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)

            print('Starting validation')
            # Evaluation for this fold
            model.eval()
            val_start_time = time.time()
            correct, total = 0, 0
            with torch.no_grad():

                for i, data in enumerate(val_loader, 0):
                    # Get inputs
                    inputs, targets = data

                    # Generate outputs
                    outputs = model(inputs)

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
                print("-------------------------------") 
                results[fold] = 100.0 * (correct / total)
                val_end_time = time.time()
                writer.add_scalar('CV Validation time', val_end_time-val_start_time, fold+1)

        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print("-------------------------------") 
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
            writer.add_scalar('CV validation accuracy', value, key+1)
        print(f'Average: {sum / len(results.items())} %')
        writer.add_scalar('CV AVG Validation accuracy', sum / len(results.items()), 0)



    else: #Holdout method
        print("-------------------------------")
        print("Dataset Information:")
        print("-------------------------------")  
        validation_split = 0.2
        shuffle_dataset = False
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        print('Dataset size: ', dataset_size)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        print('Train size: ', len(train_indices))
        #print(train_indices)
        print('Validation size: ', len(val_indices))
        #print(val_indices) 
        print("-------------------------------") 
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory= args.pin_memory, num_workers=args.num_workers)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=args.pin_memory, num_workers=args.num_workers)
        
        #Create Grid of images to show the transformation
        #dataiter = iter(train_loader)
        #videos, labels = dataiter.next()
        #vid1 = videos[0]
        #vid2 = videos[16]
        #vid1 = vid1.permute(1,0,2,3)
        #vid2 = vid2.permute(1,0,2,3)
        #img_grid1 = torchvision.utils.make_grid(vid1)
        #img_grid2 = torchvision.utils.make_grid(vid2)
        #writer.add_image('seq example 1', img_grid1)
        #writer.add_image('seq example 2', img_grid2)

        
        train_loader = CPU_GPU.DeviceDataLoader(train_loader, device)
        val_loader = CPU_GPU.DeviceDataLoader(val_loader, device)


        print("-------------------------------") 
        print("Training Starts")
        print("-------------------------------")
        model.train()
        tr_start_time = time.time()
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/slowfast/'),
            record_shapes=True
        ) as prof:
            for epoch in range(args.num_epochs):
                running_loss = 0.0
                ep_start_time = time.time()
                for i, data in enumerate(train_loader, 0):
                    # Get the inputs; data is a list of [inputs, labels]
                    inputs, targets = data
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward + Backward + Optimize
                    outputs = model(inputs)

                    loss = criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()

                
                    running_loss += loss.item()
                    #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
                print(f'Epoch: {epoch+1} finished')
                ep_end_time = time.time()
                writer.add_scalar('Epochs time', ep_end_time-ep_start_time, epoch+1)
                writer.add_scalar('Training loss', running_loss, epoch+1)
                prof.step()
            print('Finished Training')

        tr_end_time = time.time()
        print("*** Training time: ",tr_end_time-tr_start_time," ***")
        writer.add_scalar('Training time', tr_end_time-tr_start_time, 0)
        print("-------------------------------") 
        torch.save(model.state_dict(),f"{save_directory}{model_name}.pth")

        print("-------------------------------") 
        print("Validation Starts")
        print("-------------------------------") 
        model.eval()
        val_start_time = time.time()
        correct, total = 0, 0
        conf_mat = torch.zeros(args.num_categories, args.num_categories)
        with torch.no_grad():
            for i, data in enumerate(val_loader,0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, targets = data

                outputs = model(inputs)

                #probs = F.softmax(outputs, dim= 1)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    conf_mat[t.long(), p.long()] += 1

        val_end_time = time.time()
        print("Confusion matrix: ")
        print(conf_mat)
        fig = utils.plot_confusion_matrix(conf_mat, dataset.get_label_mapping())
        writer.add_figure('Confusion matrix', fig)
        
        print("Per class accuracy: ")
        print(conf_mat.diag()/conf_mat.sum(1))

        val_accuracy = 100*correct / total
        print('Validation accuracy: ',val_accuracy,'%')
        writer.add_scalar('Validation accuracy', val_accuracy, 0)
        
        print("*** Validation time: ",val_end_time-val_start_time," ***")
        writer.add_scalar('Validation time', val_end_time-val_start_time, 0)
        print("-------------------------------")
    

print("-------------------------------") 
print("Memory used in Mb")
print("-------------------------------") 
mem_used_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
print(mem_used_mb)
writer.add_scalar('Memory Used in Mb', mem_used_mb, 0)
print("-------------------------------")

prog_end_time = time.time()
print("*** Program elapsed time: ", prog_end_time-prog_start_time, " ***")
writer.add_scalar('Program time', prog_end_time-prog_start_time, 0)
writer.close()
