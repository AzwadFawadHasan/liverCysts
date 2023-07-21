import multiprocessing
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

def main():
    learn = unet_learner(dls,
                         resnet50,
                         loss_func=CrossEntropyLossFlat(axis=1),
                         metrics=[foreground_acc, cust_foreground_acc])

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
