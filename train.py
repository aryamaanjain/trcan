'''
Training code for
TRCAN (model for Terrain Super-Resolution)

Introduced in the paper
Adaptive & Multi-Resolution Procedural Infinite Terrain Generation with Diffusion Models and Perlin Noise
Aryamaan Jain, Avinash Sharma and K S Rajan
'''


## Imports and setup ##

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import glob


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, flush=True)
print(torch.cuda.device_count(), flush=True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

torch.manual_seed(0)
np.random.seed(0)


## Dataset setup ##

class SRDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.fnames_hr = glob.glob('/path_to_dataset/HR/*')
        self.fnames_lr = glob.glob('/path_to_dataset/LR/*')
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # Add more here
        ])
        
    def __len__(self):
        return len(self.fnames_hr)
    
    def __getitem__(self, index):
        hr = (np.load(self.fnames_hr[index]))[None,:,:]
        lr = (np.load(self.fnames_lr[index]))[None,:,:]
        hr = torch.tensor(hr)
        lr = torch.tensor(lr)
        return lr, hr



train_data = SRDataset()
batch_train = 4
batch_valid = 4
valid_size = 0.1
num = train_data.__len__()

# Dividing the indices for train and cross validation
indices = list(range(num))
np.random.shuffle(indices)
split = int(np.floor(valid_size*num))
train_idx,valid_idx = indices[split:], indices[:split]

#Create Samplers
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(train_data, batch_size=batch_train, sampler=train_sampler, num_workers=32)
valid_loader = DataLoader(train_data, batch_size=batch_valid, sampler=valid_sampler, num_workers=32)



## TRCAN Model ##

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


def make_model(args, parent=False):
    return RCAN(args)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        act = nn.ReLU(True)
        
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_tail = [conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x 


class Args():
    def __init__(self):
        self.n_resgroups = 8
        self.n_resblocks = 16
        self.n_feats = 64
        self.reduction  = 16 
        self.scale = 8
        self.n_colors = 1
        self.res_scale = 1
    
args = Args()
model = (nn.DataParallel(make_model(args))).to(device)



## Training and validation ##

error = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)


n_epochs = 256
valid_loss_min = np.Inf
train_loss_min = np.Inf

train_losses = []
valid_losses = []

for epoch in range(n_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    
    model.train()
    for i, images in enumerate(train_loader):
        data = images[0].to(device)
        target = images[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = error(output,target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        print('Epoch %3d | Batch %6d | Train loss %18.8f' % (epoch, i, loss.item()), flush=True)
    
    model.eval()
    with torch.no_grad():
        for i, images in enumerate(valid_loader):
            data = images[0].to(device)
            target = images[1].to(device)
            output = model(data)
            loss = error(output, target)
            valid_loss += loss.item()*data.size(0)
            print('Epoch %3d | Batch %6d | Valid loss %18.8f' % (epoch, i, loss.item()), flush=True)
    
    train_loss /= len(train_loader.sampler)
    valid_loss /= len(valid_loader.sampler)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    scheduler.step()
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss), flush=True)
    
    model_save_filename = 'model_epoch_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_save_filename)
    if valid_loss <= valid_loss_min:
        print("Validation Loss decreased {:0.6f} -> {:0.6f}".format(valid_loss_min,valid_loss), flush=True)
        valid_loss_min = valid_loss
        torch.save(model.state_dict(), 'best_model_so_far.pth')


np.save('train_losses.npy', np.array(train_losses))
np.save('valid_losses.npy', np.array(valid_losses))

