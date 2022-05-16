import os
import random
import cv2
import json
import glob
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
import functools
from detectron2.structures import BoxMode

from util import get_class_names

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

# adapted from https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def split_data(file_list):
    random.seed(0)
    random.shuffle(file_list)

    # split into train and validation sets
    split = len(file_list)*90//100
    train_list = file_list[:split]
    val_list = file_list[split:]
    return train_list, val_list

def get_id(filename):
    tokens = filename.split('/')
    return tokens[4] + '_' + '_'.join(tokens[-1].split('_')[0:2])

def get_record(meta_df, category_ids, filename):
    img_id = get_id(filename)
    annos = meta_df[meta_df['id'] == img_id]
    if annos.shape[0] == 0:
        return None

    record = {}
    height, width = cv2.imread(filename).shape[:2]
    record['file_name'] = '/'.join(filename.split('/')[3:])
    record['image_id'] = img_id
    record['height'] = height
    record['width'] = width

    objs = []
    for anno_id, row in annos.iterrows():
        anno = row['segmentation']
        mask = rle_decode(anno, (height, width))
        ys, xs = np.where(mask)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        w, h = x2 - x1, y2 - y1
        rle = binary_mask_to_rle(mask)
        obj = {
            'bbox': [int(x1), int(y1), int(w), int(h)],
            'bbox_mode': BoxMode.XYWH_ABS,
            'segmentation': rle,
            'category_id': category_ids[row['class']],
            'area': int(np.sum(mask)),
            'iscrowd': 0,
            'image_id': img_id,
            'id': anno_id
        }
        objs.append(obj)
    record['annotations'] = objs
    return record

def get_dataset_dicts(input_dir, set_name, img_dir, meta_file):
    datadicts_json = f'{set_name}-dicts.json'
    if os.path.exists(datadicts_json):
        dataset_dicts = json.load(open(datadicts_json))
        print(f'loaded cached metadata for {set_name}. {len(dataset_dicts)} records')
        return dataset_dicts

    print('preparing dataset dicts...')
    all_img_files = []
    subdir = ''
    while len(all_img_files) == 0:
        all_img_files = sorted(glob.glob(f'{img_dir}/{subdir}*.png'))
        subdir += '*/'
    train_files, val_files = split_data(all_img_files)
    img_files = val_files if set_name == 'val' else train_files
    print(f'{set_name}: {len(img_files)} samples')
    meta_df = pd.read_csv(meta_file)
    category_names = get_class_names(meta_df)
    category_ids = {name: ident for ident, name in enumerate(category_names, 1)}
    meta_df.dropna(inplace=True)
    pool = mp.Pool(processes=mp.cpu_count())
    rec_func = functools.partial(get_record, meta_df, category_ids)
    dataset_dicts = pool.map(rec_func, img_files)
    pool.close()
    # filter Nones
    dataset_dicts = [d for d in dataset_dicts if d]

    print(f'loaded metadata for {set_name}. {len(dataset_dicts)} records')
    with open(datadicts_json, 'w') as fd:
        json.dump(dataset_dicts, fd, indent=4)
    return dataset_dicts

def convert_to_coco(set_name, dataset_dicts, categories):
    output_file = f'{set_name}-dicts-coco.json'
    if os.path.exists(output_file):
        print(f'{output_file} exists - skipping COCO prep')
        return
    images = []
    annotations = []
    for record in dataset_dicts:
        images.append({
            'id': record['image_id'],
            'width': record['width'], 'height': record['height'],
            'file_name': record['file_name']
        })
        annotations.extend(record['annotations'])

    anno_id = 0
    for anno in annotations:
        anno['id'] = anno_id
        anno_id += 1

    json_data =  {'categories': categories, 'images': images, 'annotations': annotations}
    with open(output_file, 'w') as fd:
        json.dump(json_data, fd, indent=4)
    print(f'{set_name} data saved to {output_file}: {len(images)} images')

def coco_prep(input_dir, set_name, img_dir, meta_file, categories):
    dataset_dicts = get_dataset_dicts(input_dir, set_name, img_dir, meta_file)
    convert_to_coco(set_name, dataset_dicts, categories)
