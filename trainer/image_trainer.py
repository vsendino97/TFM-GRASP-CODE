import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, plot_confusion_matrix
import model.metric as module_metric

class ImageTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            #for seq in range(data.size(0))):
            data = data.permute(0,2,1,3,4)
            data = torch.flatten(data, start_dim=0,end_dim=1)
            tt = torch.zeros(data.size(0),dtype=torch.long)
            for x in range(target.size(0)):
                for y in range(8):
                    tt[x*8+y] = target[x]
            data, target = data.to(self.device), tt.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
        
            if(epoch == 1 and batch_idx == 0):
                #Video input
                if (len(data.shape) == 5):
                    self.writer.add_video('input', data.cpu().permute(0,2,1,3,4))
                #Image input
                elif (len(data.shape) == 4):
                    self.writer.add_image('input', make_grid(data.cpu(), normalize=True))
            
            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        conf_mat = None
        if self.do_validation:
            val_log,conf_mat = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
       
        
        for k,v in log.items():
            if "val" in k:
                self.writer.set_step(epoch, 'valid')
            else:
                self.writer.set_step(epoch)
            self.writer.add_scalar(k.replace("val_","")+'_epoch',v)
               
        if conf_mat is not None:
            fig = plot_confusion_matrix(conf_mat, self.data_loader.dataset.get_label_mapping())
            self.writer.add_figure('Confusion matrix', fig)
            precision = torch.nan_to_num(conf_mat.diag()/conf_mat.sum(0))
            recall = torch.nan_to_num(conf_mat.diag()/conf_mat.sum(1))
            f1 = module_metric.f1_score(precision, recall)
            log.update(
                    {"conf mat": conf_mat,
                    "class prec": precision,
                    "class rec": recall,
                    "f1": f1,
                    "avg prec": sum(precision) / len(precision),
                    "avg rec": sum(recall) / len(recall),
                    "avg f1": sum(f1) / len(f1)
                    })

        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        num_classes = self.config['arch']['args']['num_categories']
        conf_mat = torch.zeros(num_classes,num_classes)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data = data.permute(0,2,1,3,4)
                data = torch.flatten(data, start_dim=0,end_dim=1)
                tt = torch.zeros(data.size(0),dtype=torch.long)
                for x in range(target.size(0)):
                    for y in range(8):
                        tt[x*8+y] = target[x]
                data, target = data.to(self.device), tt.to(self.device)


                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                module_metric.conf_mat(output, target, conf_mat)

                if(epoch == 1 and batch_idx == 0):
                    #Video input
                    if (len(data.shape) == 5):
                        self.writer.add_video('input', data.cpu().permute(0,2,1,3,4))
                    #Image input
                    elif (len(data.shape) == 4):
                        self.writer.add_image('input', make_grid(data.cpu(), normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), conf_mat

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
