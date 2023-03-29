'''
Testing code for
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
model.load_state_dict(torch.load('/path_to_model_weight.pth'))
model.eval()


## Testing ##

def calculate_psnr(img1, img2):
    pick_factor = np.max(img1) - np.min(img1)
    rmse = np.sqrt(np.mean(np.square(img1 - img2)))
    if rmse == 0:
        return float('inf')
    return 20 * np.log10((pick_factor) / rmse)


test_fnames = ['pyrenees/bassiero', 'pyrenees/forcanada', 'tyrol/durrenstein', 'tyrol/montemagro']
ts = 256
skips = [256, 128, 64, 32, 16, 8, 4, 2, 1]
ones_array = np.ones((ts,ts))

for skip in skips:
    for test_fname in test_fnames:
        bl = np.loadtxt('/path_to_dataset/'+ test_fname +'_15m.dem', dtype=np.float32, delimiter=',')
        bh = np.loadtxt('/path_to_dataset/'+ test_fname +'_2m.dem', dtype=np.float32, delimiter=',')
        csum = np.zeros(bh.shape)
        br = np.zeros(bh.shape)
        
        with torch.no_grad():
            for i in range(0,bl.shape[0]-ts+1,skip):
                for j in range(0,bl.shape[1]-ts+1,skip):
                    print('\r', i, j, end='')
                    tile_l = bl[None,None, i:i+ts, j:j+ts].copy()
                    tile_mean = np.mean(tile_l)
                    tile_l -= tile_mean
                    tile_l = torch.tensor(tile_l).to(device)
                    tile_r = model(tile_l)
                    tile_r =  tile_r.detach().cpu().numpy().squeeze()
                    tile_r += tile_mean
                    br[i:i+ts, j:j+ts] += tile_r
                    csum[i:i+ts, j:j+ts] += ones_array
                    
            for i in range(0,bl.shape[0]-ts+1,skip):
                j = bl.shape[1] - ts
                tile_l = bl[None,None, i:i+ts, j:j+ts].copy()
                tile_mean = np.mean(tile_l)
                tile_l -= tile_mean
                tile_l = torch.tensor(tile_l).to(device)
                tile_r = model(tile_l)
                tile_r =  tile_r.detach().cpu().numpy().squeeze()
                tile_r += tile_mean
                br[i:i+ts, j:j+ts] += tile_r
                csum[i:i+ts, j:j+ts] += ones_array
            
            for j in range(0,bl.shape[1]-ts+1,skip):
                i = bl.shape[0] - ts
                tile_l = bl[None,None, i:i+ts, j:j+ts].copy()
                tile_mean = np.mean(tile_l)
                tile_l -= tile_mean
                tile_l = torch.tensor(tile_l).to(device)
                tile_r = model(tile_l)
                tile_r =  tile_r.detach().cpu().numpy().squeeze()
                tile_r += tile_mean
                br[i:i+ts, j:j+ts] += tile_r
                csum[i:i+ts, j:j+ts] += ones_array
                
            i = bl.shape[0] - ts
            j = bl.shape[1] - ts
            tile_l = bl[None,None, i:i+ts, j:j+ts].copy()
            tile_mean = np.mean(tile_l)
            tile_l -= tile_mean
            tile_l = torch.tensor(tile_l).to(device)
            tile_r = model(tile_l)
            tile_r =  tile_r.detach().cpu().numpy().squeeze()
            tile_r += tile_mean
            br[i:i+ts, j:j+ts] += tile_r
            csum[i:i+ts, j:j+ts] += ones_array
            
        br /= csum
        
        rmse = np.sqrt(np.mean(np.square(bh - br)))
        psnr = calculate_psnr(bh, br)
        print('\r %4d %20s %10.5f %10.5f' % (skip, test_fname, rmse, psnr), flush=True)



