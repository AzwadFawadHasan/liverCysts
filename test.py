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
from fastai.vision.all import *

import nibabel as nib
import numpy as np
import os
import pandas as pd

import os
import glob

import PIL
import cv2
import imageio
import glob
import cv2
import imageio
from fastai.vision.all import *
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
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

import os
import glob
import cv2
import imageio

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image

# Create a meta file for nii files processing
path1 = 'LiverSegmentImages'
path2 = 'LiverSegmentImages'
file_list = []
for dirname, _, filenames in os.walk(path1):
    for filename in filenames:
        file_list.append((dirname, filename))



df_files = pd.DataFrame(file_list, columns =['dirname', 'filename'])
df_files.sort_values(by=['filename'], ascending=True)

# Map CT scan and label

df_files["mask_dirname"] = ""
df_files["mask_filename"] = ""

for i in range(131):
    ct = f"volume-{i}.nii"
    mask = f"segmentation-{i}.nii"

    df_files.loc[df_files['filename'] == ct, 'mask_filename'] = mask
    df_files.loc[df_files['filename'] == ct, 'mask_dirname'] = "LiverSegmentImages"

# drop segment rows
df_files = df_files[df_files.mask_filename != ''].sort_values(by=['filename']).reset_index(drop=True)

#print(df_files.head())

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)



sample = 0
# The following lines try to read the CT scan and mask using the file paths from the 'df_files' DataFrame.
# Check if the file paths are correct and the files are present in the specified directories.
sample_ct = read_nii(df_files.loc[sample,'dirname']+"/"+df_files.loc[sample,'filename'])
sample_mask = read_nii(df_files.loc[sample,'mask_dirname']+"/"+df_files.loc[sample,'mask_filename'])

print(f'CT Shape:   {sample_ct.shape}\nMask Shape: {sample_mask.shape}')
print(np.amin(sample_ct), np.amax(sample_ct))
print(np.amin(sample_mask), np.amax(sample_mask))

# Preprocess the nii file
# Source https://docs.fast.ai/medical.imaging

dicom_windows = types.SimpleNamespace(
    brain=(80,40),
    subdural=(254,100),
    stroke=(8,32),
    brain_bone=(2800,600),
    brain_soft=(375,40),
    lungs=(1500,-600),
    mediastinum=(350,50),
    abdomen_soft=(400,50),
    liver=(150,30),
    spine_soft=(250,50),
    spine_bone=(1800,400),
    custom = (200,60)
)

@patch
def windowed(self:Tensor, w, l):
    px = self.clone()
    px_min = l - w//2
    px_max = l + w//2
    px[px<px_min] = px_min
    px[px>px_max] = px_max
    return (px-px_min) / (px_max-px_min)

figure(figsize=(8, 6), dpi=100)

plt.imshow(tensor(sample_ct[..., 55].astype(np.float32)).windowed(*dicom_windows.liver), cmap=plt.cm.bone);


def plot_sample(array_list, color_map='nipy_spectral'):
    '''
    Plots and a slice with all available annotations
    '''
    fig = plt.figure(figsize=(20, 16), dpi=100)

    plt.subplot(1, 4, 1)
    plt.imshow(array_list[0], cmap='bone')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(tensor(array_list[0].astype(np.float32)).windowed(*dicom_windows.liver), cmap='bone');
    plt.title('Windowed Image')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Liver & Mask')
    plt.axis('off')

    #plt.show()

sample = 55

sample_slice = tensor(sample_ct[...,sample].astype(np.float32))

plot_sample([sample_ct[..., sample],
             sample_mask[..., sample]])

# Check the mask values
mask = Image.fromarray(sample_mask[...,sample].astype('uint8'), mode="L")
unique, counts = np.unique(mask, return_counts=True)
print(np.array((unique, counts)).T)


# Preprocessing functions
# Source https://docs.fast.ai/medical.imaging

class TensorCTScan(TensorImageBW): _show_args = {'cmap': 'bone'}


@patch
def freqhist_bins(self: Tensor, n_bins=100):
    "A function to split the range of pixel values into groups, such that each group has around the same number of pixels"
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float() / n_bins + (1 / 2 / n_bins),
                   tensor([0.999])])
    t = (len(imsd) * t).long()
    return imsd[t].unique()


@patch
def hist_scaled(self: Tensor, brks=None):
    "Scales a tensor using `freqhist_bins` to values between 0 and 1"
    if self.device.type == 'cuda': return self.hist_scaled_pt(brks)
    if brks is None: brks = self.freqhist_bins()
    ys = np.linspace(0., 1., len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0., 1.)


@patch
def to_nchan(x: Tensor, wins, bins=None):
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins, int) or bins != 0: res.append(x.hist_scaled(bins).clamp(0, 1))
    dim = [0, 1][x.dim() == 3]
    return TensorCTScan(torch.stack(res, dim=dim))


@patch
def save_jpg(x: (Tensor), path, wins, bins=None, quality=120):
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins, bins) * 255).byte()
    im = Image.fromarray(x.permute(1, 2, 0).numpy(), mode=['RGB', 'CMYK'][x.shape[0] == 4])
    im.save(fn, quality=quality)


_, axs = subplots(1, 1)

#sample_slice.save_jpg('test.jpg', [dicom_windows.liver, dicom_windows.custom])
# show_image(Image.open('test.jpg'), ax=axs[0], figsize=(8, 6))
#
# # Make custom JPG files for Unet training
# # Total number of 131 nii files contains 67072 slices
#
# GENERATE_JPG_FILES = True
#
# if (GENERATE_JPG_FILES):
#
#     path = Path(".")
#
#     os.makedirs('train_images', exist_ok=True)
#     os.makedirs('train_masks', exist_ok=True)
#
#     for ii in tqdm(range(0, len(df_files), 3)):  # take 1/3 nii files for training
#         curr_ct = read_nii(df_files.loc[ii, 'dirname'] + "/" + df_files.loc[ii, 'filename'])
#         curr_mask = read_nii(df_files.loc[ii, 'mask_dirname'] + "/" + df_files.loc[ii, 'mask_filename'])
#         curr_file_name = str(df_files.loc[ii, 'filename']).split('.')[0]
#         curr_dim = curr_ct.shape[2]  # 512, 512, curr_dim
#
#         for curr_slice in range(0, curr_dim, 2):  # export every 2nd slice for training
#             data = tensor(curr_ct[..., curr_slice].astype(np.float32))
#             mask = Image.fromarray(curr_mask[..., curr_slice].astype('uint8'), mode="L")
#             data.save_jpg(f"train_images/{curr_file_name}_slice_{curr_slice}.jpg",
#                           [dicom_windows.liver, dicom_windows.custom])
#             mask.save(f"train_masks/{curr_file_name}_slice_{curr_slice}_mask.png")
# else:
#     path = Path("")  # read jpg from saved kernel output





if __name__ == '__main__':
    # TRAINING
    path = Path("LiverSegmentImages")
    BATCH_SIZE = 16
    IMAGE_SIZE = 128

    codes = np.array(["background", "liver", "tumor"])


    def get_x(fname: Path):
        return fname


    def label_func(x):
        return path / 'train_masks' / f'{x.stem}_mask.png'


    tfms = [IntToFloatTensor(), Normalize()]

    db = DataBlock(blocks=(ImageBlock(), MaskBlock(codes)),  # codes = {"Backround": 0,"Liver": 1,"Tumor": 2}
                   batch_tfms=tfms,
                   splitter=RandomSplitter(),
                   item_tfms=[Resize(IMAGE_SIZE)],
                   get_items=get_image_files,
                   get_y=label_func)

    ds = db.datasets(source=path / 'train_images')

    idx = 20
    imgs = [ds[idx][0], ds[idx][1]]
    fig, axs = plt.subplots(1, 2)

    for i, ax in enumerate(axs.flatten()):
        ax.axis('off')
        ax.imshow(imgs[i])

    unique, counts = np.unique(array(ds[idx][1]), return_counts=True)

    print(np.array((unique, counts)).T)

    dls = db.dataloaders(path / 'train_images', bs=BATCH_SIZE)  # , num_workers=0
    dls.show_batch()


    def foreground_acc(inp, targ, bkg_idx=0, axis=1):  # exclude a background from metric
        "Computes non-background accuracy for multiclass segmentation"
        targ = targ.squeeze(1)
        mask = targ != bkg_idx
        return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()


    def cust_foreground_acc(inp, targ):  # # include a background into the metric
        return foreground_acc(inp=inp, targ=targ, bkg_idx=3,
                              axis=1)  # 3 is a dummy value to include the background which is 0


    learn = unet_learner(dls,
                         resnet50,
                         loss_func=CrossEntropyLossFlat(axis=1),
                         metrics=[foreground_acc, cust_foreground_acc])

    learn.fine_tune(5, wd=0.1, cbs=SaveModelCallback())

    learn.fine_tune(5, wd=0.1, cbs=SaveModelCallback())
