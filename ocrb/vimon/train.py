import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import AverageMeter

class Trainer(object):
    def __init__(self, config, model, dataloaders):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.ckpt_dir = config['trainer']['ckpt_dir']

        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader= dataloaders[1]

        ### trainings params
        self.epochs = config['trainer']['epochs']
        self.start_epoch = 0
        self.best_val_acc = np.inf
        self.best_val_epoch = 0
        self.log_loss_every = config['trainer']['log_loss_every']
        self.beta = config['trainer']['beta']
        self.gamma = config['trainer']['gamma']
        
        self.criterion_recon = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=config['optimizer']['lr'])  
        

    def train(self):            
        for epoch in range(self.start_epoch, self.epochs):
            # run epoch
            self._train_epoch(epoch)
            acc = self._evaluate(epoch) 
            
            is_best = self.best_val_acc > acc
            self.best_val_acc = np.min([self.best_val_acc, acc])
            if is_best:
                self.best_val_epoch = epoch

            # save checkpoint
            self._save_checkpoint(epoch, best=is_best)
        print('Best validation accuracy of {} in epoch {}.'.format(self.best_val_acc, self.best_val_epoch))


    def _train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        for i, data in enumerate(self.train_loader, 0):
            x = data.float().to(self.device)
            n, t, _, _, _ = x.size()

            self.optimizer.zero_grad()
            results, losses_dict = self.model(x)

            # Compute losses
            loss = losses_dict['recon_loss'] + losses_dict['pred_loss'] + \
                + self.beta * losses_dict['kl_loss'] + self.gamma * (losses_dict['curr_mask_loss'] + losses_dict['next_mask_loss'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            self.optimizer.step()
            
            losses.update(loss.detach(), n)
            
            if i % self.log_loss_every == self.log_loss_every - 1:
                print('Epoch {} [{}|{}] | Loss {:.4f}'.format(epoch, i, len(self.train_loader), losses.avg))


    def _evaluate(self, epoch):
        self.model.eval()
        recon = AverageMeter()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):
                x = data.float().to(self.device)
                n, t, _, _, _ = x.size()
                results, losses_dict = self.model(x)
                loss_mse = self.criterion_recon(results['recon_vae'], x[:,:-1]).sum(dim=[2, 3, 4]).mean()
                recon.update(loss_mse.detach(), n)

        print('Epoch {} | Eval MSE: {:.4f}'.format(epoch, recon.avg))
        return recon.avg.item()


    def _save_checkpoint(self, epoch, best=False):
        filename = 'ckpt_vimon_best.pt' if best else 'ckpt_vimon_last.pt'
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print('Save model after epoch {} as {}.'.format(epoch, filename))
