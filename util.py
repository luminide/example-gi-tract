import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LossHistory:
    def __init__(self):
        self.data = []

    def add_val_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, np.nan, val, np.nan])

    def add_train_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, val, np.nan, np.nan])

    def add_epoch_val_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, np.nan, np.nan, val])

    def save(self):
        columns = [
            'index', 'epoch', 'sample_count',
            'train_loss', 'val_loss', 'epoch_val_loss']
        df = pd.DataFrame(self.data, columns=columns)
        df.to_csv('history.csv', index=False)

def get_class_names(df):
    labels = df['class']
    return labels.unique()

def get_id(filename):
    # e.g. filename: case123_day20/scans/slice_0001_266_266_1.50_1.50.png
    # id: case123_day20_slice_0001
    tokens = filename.split('/')
    return tokens[-3] + '_' + '_'.join(tokens[-1].split('_')[:2])

# adapted from https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def get_mask(filename, meta_df, class_names):
    img_id = get_id(filename)
    annos = meta_df[meta_df['id'] == img_id].reset_index(drop=True)

    basename = os.path.basename(filename)
    tokens = basename.split('_')
    height, width = int(tokens[3]), int(tokens[2])
    mask = np.zeros((height, width, len(class_names)), dtype=np.uint8)
    class_map = {name: i for i, name in enumerate(class_names)}
    for anno_id, row in annos.iterrows():
        anno = row['segmentation']
        if pd.isnull(anno):
            continue

        class_name = row['class']
        class_id = class_map[class_name]
        mask[:, :, class_id] = rle_decode(anno, (height, width))
    return mask

def dice_coeff(labels, preds):
    return 2*(labels*preds).sum()/(labels.sum() + preds.sum() + 1e-6)

def process_files(input_dir, img_dir, meta_df, class_names):
    filename = 'train_processed.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # delete bad data
        df = df[~df['img_files'].str.contains('case7_day0') & ~df['img_files'].str.contains('case81_day30')].reset_index(drop=True)
        return df

    img_files = sorted(glob(f'{input_dir}/{img_dir}/**/*.png', recursive=True))
    mask_dir = '../masks'
    for img_file in img_files:
        mask = get_mask(img_file, meta_df, class_names)
        basename = img_file.replace(f'{input_dir}/{img_dir}/', '')
        mask_filename = f'{mask_dir}/{basename}'
        dirname = os.path.dirname(mask_filename)
        os.makedirs(dirname, exist_ok=True)
        cv2.imwrite(mask_filename, mask)
    df = pd.DataFrame()
    # delete common prefix from paths
    img_files = [f.replace(f'{input_dir}/{img_dir}/', '') for f in img_files]
    df['img_files'] = img_files
    df.to_csv(filename, index=False)
    return df

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        image = image.transpose((1, 2, 0))
        image -= image.min()
        max_val = image.max()
        if max_val != 0:
            image /= max_val
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name)
        plt.imshow(image)
    plt.show()

def make_test_augmenter(conf):
    crop_size = round(conf.image_size*conf.crop_size)
    return  A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        ToTensorV2(transpose_mask=True)
    ])
