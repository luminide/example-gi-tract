import argparse
import glob
import cv2
import numpy as np
import pandas as pd
import torch

import detectron2
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input', default='../input', metavar='DIR', help='input directory')
parser.add_argument(
    '--model_dir', default='.', type=str, metavar='PATH',
    help='path to saved models')


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

def get_masks(pred):
    instances = pred['instances']
    pred_class = torch.mode(instances.pred_classes)[0].item() + 1
    pred_masks = instances.pred_masks.cpu().numpy()

    res = []
    used = np.zeros(instances.image_size, dtype=int)
    # filter out overlaps
    for mask in pred_masks:
        mask = mask*(1 - used)
        if mask.sum() <= 0:
            continue
        used += mask
        res.append(rle_encode(mask))
    return res, pred_class

def run(input_dir, model_dir):
    meta_file = f'{input_dir}/train.csv'
    meta_df = pd.read_csv(meta_file)

    cfg = CN.load_cfg(open(f'{model_dir}/cfg.yaml'))
    print('processing test set...')
    model_file = f'{model_dir}/model.pth'
    cfg.MODEL.WEIGHTS = model_file
    print(f'loading {model_file}')

    model = build_model(cfg)
    DetectionCheckpointer(model).load(model_file)

    predictor = DefaultPredictor(cfg)
    test_names = sorted(glob.glob(f'{input_dir}/test/*.png'))
    ids = []
    masks = []
    for img_file in test_names:
        img_data = cv2.imread(img_file)
        pred = predictor(img_data)
        img_masks, pred_class = get_masks(pred)
        img_id = img_file.split('/')[-1][:-4]
        ids.extend([img_id]*len(img_masks))
        masks.extend(img_masks)

    assert len(ids) == len(masks)
    pd.DataFrame({'id': ids, 'predicted': masks}).to_csv('submission.csv', index=False)
    print(pd.read_csv('submission.csv').head())


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.input, args.model_dir)
