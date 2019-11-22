import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import *
import torchvision
import matplotlib.pyplot as plt

def getdataset(myobj):
    if myobj.dataset == 'mnist':
        return getmnist()
    if myobj.dataset == 'cifar10':
        return getcifar10()
    if myobj.dataset == 'dailymintemperatures':
        return getdailymintemperatures(myobj)

def getmnist():
    dl = DataLoader(torchvision.datasets.MNIST('/tmp/datasets/mnist', train=True, download=True))

    tensor = dl.dataset.data
    tensor = tensor.to(dtype=torch.float32)
    tr = tensor.reshape(tensor.size(0), -1) 
    tr = tr/128
    targets = dl.dataset.targets
    targets = targets.to(dtype=torch.long)

    x_train = tr[0:50000]
    y_train = targets[0:50000]
    x_valid = tr[50000:60000]
    y_valid = targets[50000:60000]

    y_valid = y_valid.to(dtype=torch.long)
    
    #bs=64

    #train_ds = TensorDataset(x_train, y_train)
    #train_dl = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)

    #valid_ds = TensorDataset(x_valid, y_valid)
    #valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    #loaders={}
    #loaders['train'] = train_dl
    #loaders['valid'] = valid_dl
    print(type(x_train), x_train.shape, x_train.type())
    print(type(y_train), y_train.shape, y_train.type())
    return x_train, y_train, x_valid, y_valid, 784, 10

def getcifar10():
    dl = DataLoader(torchvision.datasets.CIFAR10('/tmp/datasets/cifar10', train=True, download=True))

    tensor = dl.dataset.data
    print(tensor.shape)
    tensor = torch.FloatTensor(tensor)
    tensor = tensor.to(dtype=torch.float32)
    tr = tensor.reshape(tensor.size(0), -1) 
    print(tr.shape)
    tr = tr/128
    targets = dl.dataset.targets
    targets = torch.FloatTensor(targets)
    targets = targets.to(dtype=torch.long)

    x_train = tr[0:40000]
    y_train = targets[0:40000]
    x_valid = tr[40000:50000]
    y_valid = targets[40000:50000]

    y_valid = y_valid.to(dtype=torch.long)
    
    #bs=64

    #train_ds = TensorDataset(x_train, y_train)
    #train_dl = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)

    #valid_ds = TensorDataset(x_valid, y_valid)
    #valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    #loaders={}
    #loaders['train'] = train_dl
    #loaders['valid'] = valid_dl
    print(type(x_train), x_train.shape, x_train.type())
    print(type(y_train), y_train.shape, y_train.type())
    return x_train, y_train, x_valid, y_valid, 3 * 1024, 10
