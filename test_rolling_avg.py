import argparse
import torch
from tqdm import tqdm
import dataloader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import torch.nn.functional as F

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        csv_dir = config['data_loader']['args']['csv_dir'],
        data_dir = config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle= config['data_loader']['args']['shuffle'],
        validation_split= config['data_loader']['args']['validation_split'],
        training=False,
        num_workers= config['data_loader']['args']['num_workers'],
        num_frames = config['data_loader']['args']['num_frames'],
        categories= config['data_loader']['args']['categories']

    )
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    #REMOVE
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k # remove `module.`
        new_state_dict[name] = v

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(new_state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    num_classes = config['arch']['args']['num_categories']
    conf_mat = torch.zeros(num_classes,num_classes)
    num_imgs = 0
    accuracy = 0.0
    accuracy_max = 0.0
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(valid_data_loader)):
            target = target.type(torch.LongTensor)
            roll_batch = torch.zeros(data.shape[0], num_classes)
            for j in range(data.shape[0]):
                imgs = data[j,:,:,:,:].permute(1,0,2,3)
                tt = target[j]
                imgs, tt  = imgs.to(device), tt.to(device)
                output = model(imgs)
                
                #Average prediction
                roll_avg = torch.mean(output,0)
                roll_batch[j] = roll_avg

                #Max prediction
                roll_max = torch.argmax(output,1)
                max_pred = torch.mode(roll_max).values
                if(tt == max_pred):
                    accuracy_max += 1

                #Individual prediction
                num_imgs = num_imgs+1
                tt = tt.repeat(imgs.shape[0])
                accuracy += module_metric.accuracy(output,tt)
                
                #probs = F.softmax(output, dim= 1)
                #probs_avg = F.softmax(roll_avg)
                #print("Output: ", output)
                #print("Probs: ", probs)
                #print("Probs avg: ", probs_avg)
                #print("Roll avg: ", roll_avg)
                #print("Target: ", tt)
                
                
            # computing loss, metrics on test set
            loss = loss_fn(roll_batch, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(roll_batch, target) * batch_size
            module_metric.conf_mat(roll_batch, target, conf_mat)

    n_samples = len(valid_data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    log.update({
        "Individual accuracy": accuracy/num_imgs,
        "Rolling Max accuracy": accuracy_max/num_imgs
    })
    precision = torch.nan_to_num(conf_mat.diag()/conf_mat.sum(0))
    recall = torch.nan_to_num(conf_mat.diag()/conf_mat.sum(1))
    f1 = module_metric.f1_score(precision, recall)
    log.update({
        "conf mat": conf_mat,
        "class prec": precision,
        "class rec": recall,
        "f1": f1, 
        "avg prec": sum(precision) / len(precision),
        "avg rec": sum(recall) / len(recall),
        "avg f1": sum(f1) / len(f1)
        }   
    )
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
