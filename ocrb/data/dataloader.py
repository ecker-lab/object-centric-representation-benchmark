import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, mode='train', n_steps=10, dataset_class='vmds', transform=None, path=None):
        assert dataset_class in ['vmds', 'vor', 'spmot']
        self.transform = transform
        imgs = np.load(os.path.join(path, dataset_class, '{}_{}.npy'.format(dataset_class, mode)))
        imgs = imgs[:, :n_steps]
        self.num_samples = len(imgs)
        self.imgs = [img for img in imgs]

    def __getitem__(self, index):
        x = self.imgs.__getitem__(index)
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.num_samples


def build_dataloader(batch_size, use_cuda=torch.cuda.is_available(), num_workers=1, n_steps=10, dataset_class='vmds', path='./data/data'):

    kwargs = {'num_workers':num_workers, 'pin_memory':True} if use_cuda else {}
    tform = torchvision.transforms.Lambda(lambda n: n / 255.)

    train_loader = torch.utils.data.DataLoader(
            SyntheticDataset(mode='train', n_steps=n_steps, transform=tform, dataset_class=dataset_class, path=path),
            batch_size=batch_size, 
            shuffle=True, 
            **kwargs)

    val_loader = torch.utils.data.DataLoader(
            SyntheticDataset(mode='val', n_steps=n_steps, transform=tform, dataset_class=dataset_class, path=path),
            batch_size=batch_size, 
            shuffle=False,
            **kwargs)

    return train_loader, val_loader


def build_testloader(batch_size, use_cuda=torch.cuda.is_available(), num_workers=1, n_steps=10, dataset_class='vmds', path='./data/data'):

    kwargs = {'num_workers':num_workers, 'pin_memory':True} if use_cuda else {}
    tform = torchvision.transforms.Lambda(lambda n: n / 255.)

    test_loader = torch.utils.data.DataLoader(
            SyntheticDataset(mode='test', n_steps=n_steps, transform=tform, dataset_class=dataset_class, path=path),
            batch_size=batch_size, 
            shuffle=False,
            **kwargs)

    return test_loader
