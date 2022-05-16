import os
import sys
import argparse
import random
import multiprocessing as mp
import numpy as np
import pandas as pd

import torch.backends.cudnn as cudnn
import torch

from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.engine import HookBase
from detectron2.utils import comm
from detectron2.engine import PeriodicWriter
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from evaluator import MAPIOUEvaluator
from util import LossHistory, get_class_names
from augment import build_train_augmentation
from config import Config


parser = argparse.ArgumentParser()
parser.add_argument(
    '-j', '--num-workers', default=mp.cpu_count(), type=int, metavar='N',
    help='number of data loading workers')
parser.add_argument(
    '--epochs', default=40, type=int, metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--seed', default=0, type=int,
    help='seed for initializing the random number generator')
parser.add_argument(
    '--resume', default='', type=str, metavar='PATH',
    help='path to saved model')
parser.add_argument(
    '--input', default='../input', metavar='DIR',
    help='input directory')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# adapted from https://github.com/facebookresearch/detectron2/issues/810
class ValidationLossHook(HookBase):

    def __init__(self, cfg, conf, batch_count):
        super().__init__()
        self.cfg = cfg.clone()
        self.conf = conf
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self.batch_count = batch_count
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.best_loss = None
        self.epoch_losses = []
        val_len = len(DatasetCatalog.get('val'))
        val_batch_count = val_len//cfg.SOLVER.IMS_PER_BATCH
        self.val_interval = batch_count//val_batch_count
        assert self.val_interval > 0
        self.sample_count = 0

    def after_step(self):
        curr_iter = self.trainer.iter
        self.sample_count += self.cfg.SOLVER.IMS_PER_BATCH*self.cfg.DATALOADER.NUM_WORKERS
        epoch = curr_iter//self.batch_count
        # only consider mask loss
        training_loss = self.trainer.storage.latest()['loss_mask'][0]
        self.trainer.loss_history.add_train_loss(epoch, self.sample_count, training_loss)
        # validate once every val_interval minibatches
        if curr_iter % self.val_interval == 0:
            data = next(self._loader)
            with torch.no_grad():
                loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            # only consider mask loss
            val_loss = loss_dict_reduced['val_loss_mask']
            self.trainer.loss_history.add_val_loss(epoch, self.sample_count, val_loss)
            self.epoch_losses.append(val_loss)
            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    val_total_loss=val_loss, **loss_dict_reduced)

        if (curr_iter + 1) % self.batch_count == 0:
            # finished an epoch
            curr_loss = np.mean(self.epoch_losses)
            self.trainer.loss_history.add_epoch_val_loss(epoch, self.sample_count, curr_loss)
            self.trainer.loss_history.save()
            print(f'epoch {epoch} iter {curr_iter} curr loss {curr_loss:.4f}')
            is_best = self.best_loss is None or curr_loss < self.best_loss
            if is_best:
                self.best_loss = curr_loss
                additional_state = {'iteration': curr_iter, 'val_loss': curr_loss, 'conf': self.conf.as_dict()}
                self.trainer.checkpointer.save('model', **additional_state)
                print(f'Saved model at iter {curr_iter}. val_loss {curr_loss:.4f}')
            self.epoch_losses = []


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)

    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        aug = build_train_augmentation(cfg)
        mapper = DatasetMapper(cfg, is_train=True, augmentations=aug)
        return build_detection_train_loader(cfg, mapper=mapper, sampler=sampler)


def main():
    args = parser.parse_args()
    train_set = 'train'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    input_dir = args.input
    conf = Config()
    print(conf)

    img_dir = f'{input_dir}/train'
    for set_name in ['train', 'val']:
        json_file = f'{set_name}-dicts-coco.json'
        if not os.path.exists(json_file):
            print(f'{json_file} not found')
            print('error: prep.sh must be run to preprocess the dataset')
            sys.exit()
        register_coco_instances(set_name, {}, json_file, img_dir)

    meta_file = f'{input_dir}/train.csv'
    meta_df = pd.read_csv(meta_file)
    num_classes = len(get_class_names(meta_df))
    train_len = len(DatasetCatalog.get(train_set))
    cfg_name = conf.arch
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.INPUT.MIN_SIZE_TRAIN = (conf.min_size, conf.min_size + 16, conf.min_size + 32)
    cfg.INPUT.MAX_SIZE_TRAIN = conf.max_size
    cfg.INPUT.MIN_SIZE_TEST = conf.min_size
    cfg.INPUT.MAX_SIZE_TEST = conf.max_size
    cfg.INPUT.CROP = CN({"ENABLED": True})
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [conf.crop_size, conf.crop_size]
    cfg.DATASETS.TRAIN = (train_set,)
    cfg.DATASETS.TEST = ('val',)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.IMS_PER_BATCH = conf.ims_per_batch
    cfg.SOLVER.BASE_LR = conf.lr
    cfg.SOLVER.MOMENTUM = conf.momentum
    cfg.SOLVER.NESTEROV = conf.nesterov
    batch_count = train_len//cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.MAX_ITER = args.epochs*batch_count
    cfg.SOLVER.GAMMA = conf.gamma
    if conf.decay_steps == -1:
        # disable decay
        cfg.SOLVER.STEPS = []
    else:
        cfg.SOLVER.STEPS = [conf.decay_steps]
    cfg.OUTPUT_DIR = './'
    if conf.pretrained:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = conf.bsz_per_img
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf.score_thresh_test
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = conf.loss_type
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = conf.reg_loss_weight
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.FPN.OUT_CHANNELS = conf.fpn_channels
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = conf.prop_gen
    cfg.MODEL.RPN.NMS_THRESH = conf.nms_thresh
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE =  conf.loss_type
    # validate after every epoch
    cfg.TEST.EVAL_PERIOD = batch_count
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    # disable periodic checkpointing
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER + 1
    cfg.SOLVER.AMP = CN({"ENABLED": True})
    print(f'batch_count {batch_count}. {train_len} images in training set')
    with open('cfg.yaml', 'w') as fd:
        fd.write(cfg.dump())

    trainer = Trainer(cfg)
    trainer.loss_history = LossHistory()
    trainer.register_hooks([ValidationLossHook(cfg, conf, batch_count)])
    # The PeriodicWriter needs to be the last hook, otherwise it wont
    # have access to val_loss metrics
    # ref: https://github.com/facebookresearch/detectron2/issues/810
    periodic_writer_hook = [hook for hook in trainer._hooks if isinstance(hook, PeriodicWriter)]
    all_other_hooks = [hook for hook in trainer._hooks if not isinstance(hook, PeriodicWriter)]
    trainer._hooks = all_other_hooks + periodic_writer_hook

    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    trainer.loss_history.save()


if __name__ == '__main__':
    main()
