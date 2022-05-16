import numpy as np
import torch
import pycocotools.mask as mask_util

import detectron2
from detectron2.data import DatasetCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator


# adapted from https://www.kaggle.com/theoviel/competition-metric-map-iou
def precision_at(threshold, iou):
    matches = iou > threshold
    if len(matches.shape) < 2:
        return 0, 1, 0
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ, thresh):
    instances = pred['instances']
    pred_class = torch.mode(instances.pred_classes)[0].item() + 1
    pred_masks = instances.pred_masks.cpu().numpy()

    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec), pred_class

# adapted from https://www.kaggle.com/code/slawekbiel/positive-score-with-detectron-2-3-training
class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, thresh=0.5):
        self.thresh = thresh
        self.dataset_name = dataset_name
        self.dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in self.dataset_dicts}

    def reset(self):
        self.scores = []
        self.class_scores = {}

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)
            else:
                targ = self.annotations_cache[inp['image_id']]
                res, pred_class = score(out, targ, self.thresh)
                if res is not None:
                    self.scores.append(res)
                    if pred_class not in self.class_scores:
                        self.class_scores[pred_class] = []
                    self.class_scores[pred_class].append(res)

    def evaluate(self):
        cls_res = {}
        for key in self.class_scores:
            if len(self.class_scores[key]) == 0:
                mean_score = 0
            else:
                mean_score = np.mean(self.class_scores[key])
            cls_res[key] = round(mean_score, 4)
        print(f'class scores: {sorted(cls_res.items())}')
        iou = round(np.mean(self.scores), 4)
        return {f'{self.dataset_name} MaP IoU': iou}
