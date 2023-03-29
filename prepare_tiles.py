from __future__ import division
from PIL import Image
import numpy as np
import os
import sys


def variationsDEM(h, tilePrefix):
    r1 = np.rot90(h)
    r2 = np.rot90(r1)
    r3 = np.rot90(r2)
    tr = h.transpose()
    fu = np.flipud(h)
    fr = np.fliplr(h)
    
    np.save(tilePrefix + '_rot090.npy', r1)
    np.save(tilePrefix + '_rot180.npy', r2)
    np.save(tilePrefix + '_rot270.npy', r3)
    np.save(tilePrefix + '_transp.npy', r1)
    np.save(tilePrefix + '_flipud.npy', fu)
    np.save(tilePrefix + '_fliplr.npy', fr)
    
    
def variationsOrtho(img, tilePrefix):
    img.transpose(Image.ROTATE_90).save(tilePrefix + '_rot090.jpg')
    img.transpose(Image.ROTATE_180).save(tilePrefix + '_rot180.jpg')
    img.transpose(Image.ROTATE_270).save(tilePrefix + '_rot270.jpg')
    img.transpose(Image.TRANSPOSE).save(tilePrefix + '_transp.jpg')
    img.transpose(Image.FLIP_TOP_BOTTOM).save(tilePrefix + '_flipud.jpg')
    img.transpose(Image.FLIP_LEFT_RIGHT).save(tilePrefix + '_fliplr.jpg')
    
    
def tileDataset(setName, outDir, createVariations, substractMean, tileSize, orthoSize):
    
    # read set
    H2 = np.loadtxt(setName + '_2m.dem', dtype=np.float32, delimiter=',')
    H15 = np.loadtxt(setName + '_15m.dem', dtype=np.float32, delimiter=',')
    Ortho = np.array(Image.open(setName + '_ortho1m.jpg'))
        
    # split into tiles
    demW = H2.shape[1]
    demH = H2.shape[0]
    tilesX = demW//tileSize
    tilesY = demH//tileSize
    offsetX = int(np.floor(0.5*(demW - tilesX*tileSize)))
    offsetY = int(np.floor(0.5*(demH - tilesY*tileSize)))
    print(setName, '%d x %d'%(tilesX, tilesY))
    
    for ty in range(tilesY):
        for tx in range(tilesX):
                
            # dem
            xmin = tx*tileSize + offsetX
            xmax = xmin + tileSize
            ymin = ty*tileSize + offsetY
            ymax = ymin + tileSize 
            crop2m = H2[ymin:ymax, xmin:xmax]
            crop15m = H15[ymin:ymax, xmin:xmax]                        
            if substractMean:
                tileMean = np.mean(crop15m, dtype=np.float64)
                crop2m = crop2m - tileMean
                crop15m = crop15m - tileMean
                
            # orthophoto
            xmin = tx*orthoSize + 2*offsetX
            xmax = xmin + orthoSize
            ymin = ty*orthoSize + 2*offsetY
            ymax = ymin + orthoSize       
            cropOrtho = Image.fromarray(Ortho[ymin:ymax, xmin:xmax, :])
            
            # save         
            tileName = '%s_%03d_%03d' % (setName, tx, ty)
            path2m = os.path.join('HR', tileName)
            path15m = os.path.join('LR', tileName)
            pathOrtho = os.path.join('ortho', tileName)
            
            np.save(path2m + '.npy', crop2m)
            np.save(path15m + '.npy', crop15m)
            # cropOrtho.save(pathOrtho + '.jpg')
            if createVariations:
                variationsDEM(crop2m, path2m)
                variationsDEM(crop15m, path15m)
                # variationsOrtho(cropOrtho, pathOrtho)
            
           

# input arguments and constants
SET_NAME = sys.argv[1]
OUT_DIR = sys.argv[2]
AUGMENTATION = True
SUBSTRACT_MEAN = True
TILE_SIZE = 256
ORTHO_SIZE = 512           
tileDataset(SET_NAME, OUT_DIR, AUGMENTATION, SUBSTRACT_MEAN, TILE_SIZE, ORTHO_SIZE)
