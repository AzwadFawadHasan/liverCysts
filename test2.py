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

# ... (previous imports and code)
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
path = Path("LiverSegmentImages")
BATCH_SIZE = 16
IMAGE_SIZE = 128

codes = np.array(["background", "liver", "tumor"])

def get_x(fname: Path):
    return fname

def label_func(x):
    return path / 'train_masks' / f'{x.stem}_mask.png'

def cust_foreground_acc(inp, targ):  # include a background into the metric
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1)  # 3 is a dummy value to include the background which is 0

def plot_sample(images):
    fig, axs = plt.subplots(1, len(images))

    for i, ax in enumerate(axs.flatten()):
        ax.axis('off')
        ax.imshow(images[i], cmap='gray')

    plt.show()



if __name__ == '__main__':
    tfms = [IntToFloatTensor(), Normalize()]

    db = DataBlock(blocks=(ImageBlock(), MaskBlock(codes)),
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

    dls = db.dataloaders(path / 'train_images', bs=BATCH_SIZE)
    learn = unet_learner(dls,
                         resnet50,
                         loss_func=CrossEntropyLossFlat(axis=1),
                         metrics=[foreground_acc, cust_foreground_acc])

    #learn.fine_tune(5, wd=0.1, cbs=SaveModelCallback())
    #learn.show_results()
    learn.export(path / f'Liver_segmentation')
    # Load saved model
    IMAGE_SIZE = 128


    tfms = [Resize(IMAGE_SIZE), IntToFloatTensor(), Normalize()]
    learn0 = load_learner(path / f'Liver_segmentation', cpu=False)
    learn0.dls.transform = tfms

    # Load saved model
    IMAGE_SIZE = 128
    GENERATE_JPG_FILES = True
    if (GENERATE_JPG_FILES):
        tfms = [Resize(IMAGE_SIZE), IntToFloatTensor(), Normalize()]
        learn0 = load_learner(path / f'Liver_segmentation', cpu=False)
        learn0.dls.transform = tfms


    def read_nii(filepath):
        '''
        Reads .nii file and returns pixel array
        '''
        ct_scan = nib.load(filepath)
        array = ct_scan.get_fdata()
        array = np.rot90(np.array(array))
        return (array)
    def nii_tfm(fn, wins):

        test_nii = read_nii(fn)
        curr_dim = test_nii.shape[2]  # 512, 512, curr_dim
        slices = []

        for curr_slice in range(curr_dim):
            data = tensor(test_nii[..., curr_slice].astype(np.float32))
            data = (data.to_nchan(wins) * 255).byte()
            slices.append(TensorImage(data))

        return slices


    tst = 20

    test_nii = read_nii(df_files.loc[tst, 'dirname'] + "/" + df_files.loc[tst, 'filename'])
    test_mask = read_nii(df_files.loc[tst, 'mask_dirname'] + "/" + df_files.loc[tst, 'mask_filename'])
    print(test_nii.shape)

    test_slice_idx = 500

    sample_slice = tensor(test_nii[..., test_slice_idx].astype(np.float32))

    plot_sample([test_nii[..., test_slice_idx], test_mask[..., test_slice_idx]])

    # Prepare a nii test file for prediction
    dicom_windows = types.SimpleNamespace(
        brain=(80, 40),
        subdural=(254, 100),
        stroke=(8, 32),
        brain_bone=(2800, 600),
        brain_soft=(375, 40),
        lungs=(1500, -600),
        mediastinum=(350, 50),
        abdomen_soft=(400, 50),
        liver=(150, 30),
        spine_soft=(250, 50),
        spine_bone=(1800, 400),
        custom=(200, 60)
    )
    dicom_windows = types.SimpleNamespace(
        brain=(80, 40),
        subdural=(254, 100),
        stroke=(8, 32),
        brain_bone=(2800, 600),
        brain_soft=(375, 40),
        lungs=(1500, -600),
        mediastinum=(350, 50),
        abdomen_soft=(400, 50),
        liver=(150, 30),
        spine_soft=(250, 50),
        spine_bone=(1800, 400),
        custom=(200, 60)
    )


    @patch
    def windowed(self: Tensor, w, l):
        px = self.clone()
        px_min = l - w // 2
        px_max = l + w // 2
        px[px < px_min] = px_min
        px[px > px_max] = px_max
        return (px - px_min) / (px_max - px_min)


    figure(figsize=(8, 6), dpi=100)
    # Create a meta file for nii files processing
    path1 = 'LiverSegmentImages'
    path2 = 'LiverSegmentImages'
    file_list = []
    for dirname, _, filenames in os.walk(path1):
        for filename in filenames:
            file_list.append((dirname, filename))

    df_files = pd.DataFrame(file_list, columns=['dirname', 'filename'])
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


    # print(df_files.head())

    def read_nii(filepath):
        '''
        Reads .nii file and returns pixel array
        '''
        ct_scan = nib.load(filepath)
        array = ct_scan.get_fdata()
        array = np.rot90(np.array(array))
        return (array)


    sample = 0
    # The following lines try to read the CT scan and mask using the file paths from the 'df_files' DataFrame.
    # Check if the file paths are correct and the files are present in the specified directories.
    sample_ct = read_nii(df_files.loc[sample, 'dirname'] + "/" + df_files.loc[sample, 'filename'])
    sample_mask = read_nii(df_files.loc[sample, 'mask_dirname'] + "/" + df_files.loc[sample, 'mask_filename'])

    print(f'CT Shape:   {sample_ct.shape}\nMask Shape: {sample_mask.shape}')
    print(np.amin(sample_ct), np.amax(sample_ct))
    print(np.amin(sample_mask), np.amax(sample_mask))

    # Preprocess the nii file
    # Source https://docs.fast.ai/medical.imaging

    dicom_windows = types.SimpleNamespace(
        brain=(80, 40),
        subdural=(254, 100),
        stroke=(8, 32),
        brain_bone=(2800, 600),
        brain_soft=(375, 40),
        lungs=(1500, -600),
        mediastinum=(350, 50),
        abdomen_soft=(400, 50),
        liver=(150, 30),
        spine_soft=(250, 50),
        spine_bone=(1800, 400),
        custom=(200, 60)
    )


    @patch
    def windowed(self: Tensor, w, l):
        px = self.clone()
        px_min = l - w // 2
        px_max = l + w // 2
        px[px < px_min] = px_min
        px[px > px_max] = px_max
        return (px - px_min) / (px_max - px_min)


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

        # plt.show()


    sample = 55

    sample_slice = tensor(sample_ct[..., sample].astype(np.float32))

    plot_sample([sample_ct[..., sample],
                 sample_mask[..., sample]])

    # Check the mask values
    mask = Image.fromarray(sample_mask[..., sample].astype('uint8'), mode="L")
    unique, counts = np.unique(mask, return_counts=True)
    print(np.array((unique, counts)).T)


    # Preprocessing functions
    # Source https://docs.fast.ai/medical.imaging

    class TensorCTScan(TensorImageBW):
        _show_args = {'cmap': 'bone'}


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

    test_files = nii_tfm(df_files.loc[tst, 'dirname'] + "/" + df_files.loc[tst, 'filename'],
                         [dicom_windows.liver, dicom_windows.custom])
    print("Number of test slices: ", len(test_files))

    # Check an input for a test file
    # show_image(test_files[test_slice_idx])
    # # Get predictions for a Test file
    #
    # test_dl = learn0.dls.test_dl(test_files)
    # preds, y = learn0.get_preds(dl=test_dl)
    #
    # predicted_mask = np.argmax(preds, axis=1)
    # print("in the last line")
    # plt.imshow(predicted_mask[test_slice_idx])
    # print("in the last 2 line")
    # plt.show()

    # Assuming you have a separate validation set with index 'val_idx'
    # val_files = [nii_tfm(df_files.loc[idx, 'dirname'] + "/" + df_files.loc[idx, 'filename'],
    #                      [dicom_windows.liver, dicom_windows.custom])
    #              for idx in val_idx]
    #
    # # Create DataLoader for validation data
    # val_dl = learn0.dls.test_dl(val_files)
    # learn = load_learner(path / f'Liver_segmentation', cpu=False)
    # preds, y = learn.get_preds(dl=val_dl)
    # Define the dice function
    def nii_tfm(fn, wins):
        test_nii = read_nii(fn)
        curr_dim = test_nii.shape[2]
        slices = []

        for curr_slice in range(curr_dim):
            data = tensor(test_nii[..., curr_slice].astype(np.float32))
            data = (data.to_nchan(wins) * 255).byte()
            slices.append(TensorImage(data))

        # Get the ground truth mask for the corresponding CT scan
        mask_file = f"volume-{os.path.basename(fn).split('-')[1]}"  # Update the mask file name
        mask_nii = read_nii(os.path.join(os.path.dirname(fn), mask_file))
        mask_slices = []

        for curr_slice in range(curr_dim):
            mask_data = tensor(mask_nii[..., curr_slice].astype(np.float32))
            mask_data = (mask_data.to_nchan(wins) * 255).byte()
            mask_slices.append(TensorMask(mask_data))  # Use TensorMask for masks

        return slices, mask_slices  # Return the mask_slices list directly


    # Return the first element, which contains the mask slices


    def dice(preds, targs, class_id=1):
        # Assuming the class of interest is class_id=1 (change it if needed)
        p = (preds[:, class_id] > 0).float()
        t = (targs[:, class_id] > 0).float()
        intersect = (p * t).sum()
        union = p.sum() + t.sum()
        dice_score = (2.0 * intersect) / (union + 1e-8)
        return dice_score
    from pathlib import Path

    # ... (previous code)

    # Step 1: Create a DataLoader for the validation data
    validation_path = Path('LiverSegmentImages/volume_pt1')
    val_idx = list(range(len(df_files)))  # Use all samples in the DataFrame

    # Assuming you have a separate validation set with index 'val_idx'
    val_files = [nii_tfm(df_files.loc[idx, 'dirname'] + "/" + df_files.loc[idx, 'filename'],
                         [dicom_windows.liver, dicom_windows.custom])
                 for idx in val_idx]

    # Step 2: Calculate evaluation metrics
    dice_scores = []
    iou_scores = []
    foreground_accuracies = []

    for val_file, mask_file in val_files:
        # Create DataLoader for the current validation data item (single slice)
        val_dl = learn0.dls.test_dl(val_file)  # Pass the CT scan slices directly

        # Get the ground truth masks as a list of tensors (mask slices)
        y = [mask for _, mask in mask_file]

        preds, _ = learn0.get_preds(dl=val_dl)

        # Calculate evaluation metrics for the current slice
        dice_score = dice(preds, y)
        iou_score = iou(preds, y)
        foreground_accuracy = cust_foreground_acc(preds, y)

        dice_scores.append(dice_score)
        iou_scores.append(iou_score)
        foreground_accuracies.append(foreground_accuracy)

    # Average the metrics over all slices
    mean_dice_score = np.mean(dice_scores)
    mean_iou_score = np.mean(iou_scores)
    mean_foreground_accuracy = np.mean(foreground_accuracies)

    print("Mean Dice Score:", mean_dice_score)
    print("Mean IoU Score:", mean_iou_score)
    print("Mean Foreground Accuracy:", mean_foreground_accuracy)

