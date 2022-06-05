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

def create_model(model_file, num_classes):
    checkpoint = torch.load(model_file, map_location=device)
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

def pad_mask(conf, mask, num_classes):
    # pad image to conf.image_size
    padded = np.zeros((num_classes, conf.image_size, conf.image_size), dtype=mask.dtype)
    _, height, width = mask.shape
    dh = conf.image_size - height
    dw = conf.image_size - width

    top = dh//2
    left = dw//2
    padded[:, top:top + height, left:left + width] = mask
    return padded

def resize_mask(mask, height, width):
    mask = mask.transpose((1, 2, 0))
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.transpose((2, 0, 1))

def test(confs, loaders, models, img_files, class_names, thresh):
    ids = []
    classes = []
    masks = []
    sigmoid = nn.Sigmoid()
    num_classes = len(class_names)
    for model in models:
        model.eval()
    iters = [iter(loader) for loader in loaders]

    with torch.no_grad():
        for img_idx, img_file in enumerate(img_files):
            height, width = get_img_shape(img_file)
            img_id = get_id(img_file)
            for i, it in enumerate(iters):
                conf = confs[i]
                model = models[i]
                images, _ = it.next()
                images = images.to(device)
                outputs = model(images)
                pred = sigmoid(outputs).cpu().numpy()[0]
                pred = pad_mask(conf, pred, num_classes)
                pred = resize_mask(pred, height, width)
                if i == 0:
                    mean_pred = pred
                else:
                    mean_pred += pred
            mean_pred /= len(iters)
            for class_id, class_name in enumerate(class_names):
                mask = mean_pred[class_id]
                mask[mask >= thresh] = 1
                mask[mask < thresh] = 0
                enc_mask = '' if mask.sum() == 0 else rle_encode(mask)
                ids.append(img_id)
                classes.append(class_name)
                masks.append(enc_mask)
    return ids, classes, masks

def save_results(input_dir, ids, classes, masks):
    pred_df = pd.DataFrame({'id': ids, 'class': classes, 'predicted': masks})
    subm = pd.read_csv(f'{input_dir}/sample_submission.csv')
    del subm['predicted']

    if pred_df.shape[0] > 0:
        # sort according to the given order and save to a csv file
        subm = subm.merge(pred_df, on=['id', 'class'])
    subm.to_csv('submission.csv', index=False)

def run(input_dir, model_files, thresh):
    meta_file = os.path.join(input_dir, 'train.csv')
    train_df = pd.read_csv(meta_file, dtype=str)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)
    batch_size = 1

    models = []
    confs = []
    loaders = []
    for i, model_file in enumerate(model_files):
        print(model_file)
        model, conf = create_model(model_file, num_classes)
        print(conf)
        conf['batch_size'] = batch_size
        loader, df = create_test_loader(conf, input_dir, class_names)
        models.append(model)
        confs.append(conf)
        loaders.append(loader)
    img_files = df['img_files']
    # average predictions from multiple models
    ids, classes, masks = test(confs, loaders, models, img_files, class_names, thresh)
    save_results(input_dir, ids, classes, masks)

if __name__ == '__main__':
    test_thresh = 0.5
    model_dir = './'
    model_files = sorted(glob(f'{model_dir}/model*.pth'))
    run('../input', model_files, test_thresh)
