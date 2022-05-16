from detectron2.data import transforms as T

from config import Config

def build_train_augmentation(cfg):
    conf = Config()
    min_size = cfg.INPUT.MIN_SIZE_TRAIN
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    augmentation = [
        T.RandomApply(
            T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            prob=conf.aug_prob*0.5
        ),
        T.RandomApply(
            T.RandomRotation([-conf.angle, conf.angle]),
                prob=conf.aug_prob*0.2),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomApply(T.RandomBrightness(0.9, 1.1), prob=conf.aug_prob*0.2),
        T.RandomApply(T.RandomContrast(0.9, 1.1), prob=conf.aug_prob*0.2),
        T.ResizeShortestEdge(min_size, max_size, sample_style),
    ]
    return augmentation


