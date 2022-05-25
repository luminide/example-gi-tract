import os
import cv2
from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data

from util import get_class_names, make_test_augmenter, get_id
from dataset import VisionDataset
from models import ModelWrapper
from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

def create_test_loader(conf, input_dir, class_names):
    test_aug = make_test_augmenter(conf)
    test_df = pd.DataFrame()
    img_files = []
    img_dir = 'test'
    subdir = ''
    while len(img_files) == 0 and len(subdir) < 10:
        img_files = sorted(glob(f'{input_dir}/{img_dir}/{subdir}*.png'))
        subdir += '*/'
        if len(subdir) > 10:
            return None
    # delete common prefix from paths
    img_files = [f.replace(f'{input_dir}/{img_dir}/', '') for f in img_files]

    test_df['img_files'] = img_files
    test_dataset = VisionDataset(
        test_df, conf, input_dir, img_dir,
        class_names, test_aug, is_test=True)
    print(f'{len(test_dataset)} examples in test set')
    loader = data.DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=False,
        num_workers=mp.cpu_count(), pin_memory=False)
    return loader, test_df

def create_model(model_dir, num_classes):
    checkpoint = torch.load(f'{model_dir}/model.pth', map_location=device)
    conf = Config(checkpoint['conf'])
    conf.pretrained = False
    model = ModelWrapper(conf, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    return model, conf

def rle_encode(img):
    '''
    this function is adapted from
    https://www.kaggle.com/code/stainsby/fast-tested-rle/notebook
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_img_shape(filename):
    basename = os.path.basename(filename)
    tokens = basename.split('_')
    height, width = int(tokens[3]), int(tokens[2])
    return (height, width)

def pad_mask(conf, mask):
    # pad image to conf.image_size
    padded = np.zeros((conf.image_size, conf.image_size), dtype=mask.dtype)
    dh = conf.image_size - mask.shape[0]
    dw = conf.image_size - mask.shape[1]

    top = dh//2
    left = dw//2
    padded[top:top + mask.shape[0], left:left + mask.shape[1]] = mask
    return padded

def resize_mask(mask, height, width):
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

def run(input_dir, model_dir, thresh):
    meta_file = os.path.join(input_dir, 'train.csv')
    train_df = pd.read_csv(meta_file, dtype=str)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)

    model, conf = create_model(model_dir, num_classes)
    loader, df = create_test_loader(conf, input_dir, class_names)
    img_files = df['img_files']

    subm = pd.read_csv(f'{input_dir}/sample_submission.csv')
    del subm['predicted']

    ids = []
    classes = []
    masks = []
    img_idx = 0
    sigmoid = nn.Sigmoid()
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = sigmoid(outputs).cpu().numpy()
            preds[preds >= thresh] = 1
            preds[preds < thresh] = 0
            for pred in preds:
                img_file = img_files[img_idx]
                img_idx += 1
                img_id = get_id(img_file)
                height, width = get_img_shape(img_file)
                for class_id, class_name in enumerate(class_names):
                    mask = pred[class_id]
                    mask = pad_mask(conf, mask)
                    mask = resize_mask(mask, height, width)
                    enc_mask = '' if mask.sum() == 0 else rle_encode(mask)
                    ids.append(img_id)
                    classes.append(class_name)
                    masks.append(enc_mask)

    pred_df = pd.DataFrame({'id': ids, 'class': classes, 'predicted': masks})
    if pred_df.shape[0] > 0:
        # sort according to the given order and save to a csv file
        subm = subm.merge(pred_df, on=['id', 'class'])
    subm.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    test_thresh = 0.5
    run('../input', './', test_thresh)
