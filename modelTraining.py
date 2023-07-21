import os
import glob
import cv2
import imageio

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from ipywidgets import *
from PIL import Image
from matplotlib.pyplot import figure

from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *

import numpy as np
path = Path("C:/Users/dellG15/Documents/liverCysts")

BATCH_SIZE = 16
IMAGE_SIZE = 128

codes = np.array(["background", "liver", "tumor"])


def get_x(fname: Path): return fname


def label_func(x): return path / 'train_masks' / f'{x.stem}_mask.png'


tfms = [IntToFloatTensor(), Normalize()]

db = DataBlock(blocks=(ImageBlock(), MaskBlock(codes)),  # codes = {"Backround": 0,"Liver": 1,"Tumor": 2}
               batch_tfms=tfms,
               splitter=RandomSplitter(),
               item_tfms=[Resize(IMAGE_SIZE)],
               get_items=get_image_files,
               get_y=label_func)

ds = db.datasets(source=path / 'train_images')
ds = db.datasets(source=path / 'train_images')
idx = 20
imgs = [ds[idx][0],ds[idx][1]]
fig, axs = plt.subplots(1, 2)

for i,ax in enumerate(axs.flatten()):
    ax.axis('off')
    ax.imshow(imgs[i])
    #plt.show()

unique, counts = np.unique(array(ds[idx][1]), return_counts=True)

print( np.array((unique, counts)).T)

dls = db.dataloaders(path/'train_images', bs = BATCH_SIZE) #, num_workers=0
dls.show_batch()
#plt.show()

def foreground_acc(inp, targ, bkg_idx=0, axis=1):  # exclude a background from metric
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask]==targ[mask]).float().mean()

def cust_foreground_acc(inp, targ):  # # include a background into the metric
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1) # 3 is a dummy value to include the background which is 0

import multiprocessing

def main():
    learn = unet_learner(dls,
                         resnet50,
                         loss_func=CrossEntropyLossFlat(axis=1),
                         metrics=[foreground_acc, cust_foreground_acc])
    learn.fine_tune(5, wd=0.1, cbs=SaveModelCallback())

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

