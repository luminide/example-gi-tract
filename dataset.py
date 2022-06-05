import os
import cv2
import numpy as np
import torch.utils.data as data


class VisionDataset(data.Dataset):
    def __init__(
            self, df, conf, input_dir, imgs_dir,
            class_names, transform, is_test=False, subset=100):
        self.conf = conf
        self.transform = transform
        self.is_test = is_test

        if 'num_slices' not in self.conf._params:
            self.conf['num_slices'] = 5
        if subset != 100:
            assert subset < 100
            # train and validate on subsets
            num_rows = df.shape[0]*subset//100
            df = df.iloc[:num_rows]

        files = df['img_files']
        self.files = [os.path.join(input_dir, imgs_dir, f) for f in files]
        self.masks = [os.path.join('../masks', f) for f in files]

    def resize(self, img, interp):
        return  cv2.resize(
            img, (self.conf.image_size, self.conf.image_size), interpolation=interp)

    def load_slice(self, img_file, diff):
        slice_num = os.path.basename(img_file).split('_')[1]
        filename = (
            img_file.replace(
                'slice_' + slice_num,
                'slice_' + str(int(slice_num) + diff).zfill(4)))
        if os.path.exists(filename):
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            return self.resize(img, cv2.INTER_AREA)
        return None

    def __getitem__(self, index):
        conf = self.conf
        num_slices = conf.num_slices
        assert num_slices % 2 == 1
        img_file = self.files[index]
        # read multiple slices into one image
        img = np.zeros(
            (conf.image_size, conf.image_size, num_slices), dtype=np.float32)
        for i, diff in enumerate(range(-(num_slices//2), num_slices//2 + 1)):
            slc =  self.load_slice(img_file, diff)
            if slc is None:
                continue
            img[:, :, i] = slc

        max_val = img.max()
        if max_val != 0:
            img /= max_val

        if self.is_test:
            msk = 0
            result = self.transform(image=img)
            img = result['image']
        else:
            # read mask
            msk_file = self.masks[index]
            msk = cv2.imread(msk_file, cv2.IMREAD_UNCHANGED)
            msk = self.resize(msk, cv2.INTER_NEAREST)
            msk = msk.astype(np.float32)
            result = self.transform(image=img, mask=msk)
            img, msk = result['image'], result['mask']
        return img, msk

    def __len__(self):
        return len(self.files)
