import time
import argparse

import torch

#import torchvideo

from model import VideoModel,ImageModel
from dataloader import CPU_GPU
import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the GRASP network', usage='python3 train.py [options]', epilog='The code is not very well optimized :)')
    parser.add_argument('--video_model', type=str, help='Specify the model name: r2plus1d | slow_r50 | csn | x3d_m | slowfast_r50 ', default = 'slow_r50')
    parser.add_argument('--labels_file', type=str, help='CSV file with labels', default='./augmented_dataset.csv')
    parser.add_argument('--video_path', type=str, help='Path to the folder with the videos', default='/home/victor.sendinog/data-local/transformations/')
    parser.add_argument('--batch_size', type=int, help='Batch size for the model', default=32)
    parser.add_argument('--cross_val', action='store_true', help='Specify the argument for doing cross-validation. Otherwise holdout is performed')
    parser.add_argument('--num_folds', type=int, help='Number of cross-validation folds', default=5)
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs', default=5)
    parser.add_argument('--learn_rate', type=float, help='Learning rate applied in training', default=0.001)
    parser.add_argument('--num_workers', type=int, help='Number of threads for reading the dataset', default=4)
    parser.add_argument('--pin_memory', action='store_true', help='Specify the argument to use Pytorch pin memory. By default this option is not used.')
    parser.add_argument('--save_directory', type=str, help='Path used for saving the results', default='./results')
    parser.add_argument('--model_weights', type=str, help='Path to an existing model weights file to use as initial weights', default='')
    parser.add_argument('--categories', type=str, help='Number of categories for labels. PIP | SmallCategories | BigCategories', default="BigCategories")
    parser.add_argument('--tune_layers', type=int, help='Indicate the number of convolutional layers to tune in the model', default = 7)
    parser.add_argument('--class_layer', type=str, help='Setup the architecture of the classification layer: small | full', default = 'full')
    #--video_path /home/victor.sendinog/data-local/Yale-transformations/ --num_epochs 5

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
    model_name = args.video_model + f"-{args.categories}" + time.strftime("-%Y%m%d-%H%M%S-") + str(args.num_epochs) + "ep"
    save_directory = args.save_directory + "/" + args.video_model + "/" + model_name + "/"
    print("Save directory: ", save_directory)
    print("Model weights: ", args.model_weights)
    print("Number of categories: ", args.categories)
    print("Tuning mode: ", args.tune_layers)
    print("Classification layer: ", args.class_layer)
    print("-------------------------------")

    prog_start_time = time.time()

    torch.manual_seed(42)

    num_categories = utils.get_num_categories(args.categories)

    if(args.video_model == "image"):
        model = ImageModel(args.video_model, num_categories, args.tune_layers, args.class_layer).to(device=device)

        dataset = datasets.GRASPTransformedImages(
            csv_file = args.labels_file,
            video_path = args.video_path,
            transform=torchvision.transforms.Compose([
                transforms.ReadTransformedImages(model.transform_params["num_frames"]),
                RandomCrop(244)]),
            num_frames = model.transform_params["num_frames"],
            categories = args.categories
        )
        """


    else:
        model = VideoModel(args.video_model, num_categories, args.tune_layers, args.class_layer).to(device=device)
        """

        dataset = datasets.GRASPTransformedDataset(
            csv_file = args.labels_file,
            video_path = args.video_path,
            transform=torchvision.transforms.Compose([
                transforms.ReadTransformedVideos(model.transform_params["num_frames"]),
                RandomShortSideScale(min_size=256, max_size=320),
                RandomCrop(244)]),
            num_frames = model.transform_params["num_frames"],
            categories = args.categories
        )
        """


"""

    if torch.cuda.device_count() > 1:
        print("Usage of", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    train_writer = SummaryWriter(save_directory+"train/")
    val_writer = SummaryWriter(save_directory+"val/")
    
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

        # Define K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)
        print("-------------------------------")     
        print("Cross-Validation start")
        print("-------------------------------")


        losses = utils.AverageMeter('Loss', ':.4e')
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
                losses.reset()
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
                    losses.update(loss.item(), inputs.size(0))
                    #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, current_loss))
                    
                    if i % 500 == 499:
                        print('Loss after mini-batch %5d: %.3f' %
                              (i + 1, current_loss / 500))
                        current_loss = 0.0

                    

                ep_end_time = time.time()
                train_writer.add_scalar('Fold ' + str(fold+1) + ' epochs time', ep_end_time- ep_start_time, epoch+1)
                print('Fold ' + str(fold+1) + ' epochs time', ep_end_time- ep_start_time, epoch+1)
                train_writer.add_scalar('Fold ' + str(fold+1) + ' loss', losses.avg, epoch+1)
                print('Fold ' + str(fold+1) + ' training loss', losses.avg, epoch+1)


                model.eval()
                losses.reset()
                correct, total = 0, 0
                val_start_time = time.time()
                with torch.no_grad():
                    for i, data in enumerate(val_loader,0):
                        inputs, targets = data

                        outputs = model(inputs)

                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                        loss = criterion(outputs, targets)
                        losses.update(loss.item(), inputs.size(0))


                val_end_time = time.time()
                print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
                print("-------------------------------")
                results[fold] = correct / total
                val_writer.add_scalar('Fold ' + str(fold+1) + ' epochs time', val_end_time-val_start_time, epoch+1)
                val_writer.add_scalar('Fold ' + str(fold+1) + ' loss', losses.avg, epoch+1)
            


            # Process is complete. Save the model
            print('Training CV epoch has finished. Saving trained model.')
            save_path = save_directory + f'model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)

        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print("-------------------------------") 
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
            val_writer.add_scalar('CV validation accuracy', value, key+1)
        print(f'Average: {sum / len(results.items())} %')
        val_writer.add_scalar('CV AVG Validation accuracy', sum / len(results.items()), 0)



    else: #Holdout method
        print("-------------------------------")
        print("Dataset Information:")
        print("-------------------------------")  
        validation_split = 0.2
        shuffle_dataset = True
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        print('Dataset size: ', dataset_size)
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            indices = torch.randperm(dataset_size)
        else:
            indices = list(range(dataset_size)) 
            
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
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, pin_memory=args.pin_memory, num_workers=args.num_workers)
        
        #Create Grid of images to show the transformation
        dataiter = iter(train_loader)
        videos, labels = dataiter.next()
        vid1 = videos[0]
        vid2 = videos[16]
        vid3 = videos[24]

        if(args.video_model != "image"):
            vid1 = vid1.permute(1,0,2,3)
            vid2 = vid2.permute(1,0,2,3)
            vid3 = vid3.permute(1,0,2,3) 
        img_grid1 = torchvision.utils.make_grid(vid1)
        img_grid2 = torchvision.utils.make_grid(vid2)
        img_grid3 = torchvision.utils.make_grid(vid3)
        train_writer.add_image('seq example 1', img_grid1)
        train_writer.add_image('seq example 2', img_grid2)
        train_writer.add_image('seq example 3', img_grid3)

        
        
        train_loader = CPU_GPU.DeviceDataLoader(train_loader, device)
        val_loader = CPU_GPU.DeviceDataLoader(val_loader, device)


        print("-------------------------------") 
        print("Training Starts")
        print("-------------------------------")
        conf_mat = torch.zeros(num_categories, num_categories)
        losses = utils.AverageMeter('Loss', ':.4e')
        tr_start_time = time.time()
        for epoch in range(args.num_epochs):
            model.train()
            losses.reset()
            ep_start_time = time.time()
            for i, data in enumerate(train_loader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, targets = data
                #print("inputs ",inputs.size())
                #print("target ", targets.size())
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = model(inputs)

                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                losses.update(loss.item(), inputs.size(0))
                #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            print(f'Epoch: {epoch+1} finished')
            ep_end_time = time.time()
            train_writer.add_scalar('Epochs time', ep_end_time-ep_start_time, epoch+1)
            train_writer.add_scalar('Loss', losses.avg, epoch+1)
        

            model.eval()
            losses.reset()
            correct, total = 0, 0
            val_start_time = time.time()
            with torch.no_grad():
                for i, data in enumerate(val_loader,0):
                    inputs, targets = data

                    outputs = model(inputs)

                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    loss = criterion(outputs, targets)
                    losses.update(loss.item(), inputs.size(0))

                    #Compute the confusion matrix on last epoch
                    if(epoch == (args.num_epochs-1)):
                        for t, p in zip(targets.view(-1), predicted.view(-1)):
                            conf_mat[t.long(), p.long()] += 1


            val_end_time = time.time()
            val_writer.add_scalar('Epochs time', val_end_time-val_start_time, epoch+1)
            val_writer.add_scalar('Loss', losses.avg, epoch+1)
            val_accuracy = correct / total
            val_writer.add_scalar('Validation accuracy', val_accuracy, epoch+1)
            print("-------------------------------")

        print('Finished Training')
        tr_end_time = time.time()
        print("*** Training time: ",tr_end_time-tr_start_time," ***")
        train_writer.add_scalar('Training time', tr_end_time-tr_start_time, 0)
        print("-------------------------------") 
        torch.save(model.state_dict(),f"{save_directory}{model_name}.pth")

        print("-------------------------------") 
        print("Final results")
        print("-------------------------------") 
        print("Confusion matrix: ")
        print(conf_mat)
        fig = utils.plot_confusion_matrix(conf_mat, dataset.get_label_mapping())
        val_writer.add_figure('Confusion matrix', fig)
        
        print("Per class accuracy: ")
        print(conf_mat.diag()/conf_mat.sum(1))

        val_accuracy = 100*correct / total
        print('Validation accuracy: ',val_accuracy,'%')
        
        print("-------------------------------")
        if(args.video_model == "image"):
            dataset = datasets.GRASPTransformedDataset(
                csv_file = args.labels_file,
                video_path = args.video_path,
                transform=torchvision.transforms.Compose([
                    transforms.ReadTransformedVideos(model.transform_params["num_frames"]),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244)]),
                num_frames = model.transform_params["num_frames"],
                categories = args.categories
            )

            
            val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1, sampler=val_subsampler)

            val_loader = CPU_GPU.DeviceDataLoader(val_loader, device)


    

print("-------------------------------") 
print("Memory used in Mb")
print("-------------------------------") 
mem_used_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
print(mem_used_mb)
train_writer.add_scalar('Memory Used in Mb', mem_used_mb, 0)
print("-------------------------------")

prog_end_time = time.time()
print("*** Program elapsed time: ", prog_end_time-prog_start_time, " ***")
train_writer.add_scalar('Program time', prog_end_time-prog_start_time, 0)
train_writer.close()
val_writer.close()
"""