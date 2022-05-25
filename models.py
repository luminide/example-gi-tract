import torch.nn as nn
import segmentation_models_pytorch as smp


class ModelWrapper(nn.Module):

    def __init__(self, conf, num_classes):
        super().__init__()
        if conf.arch == 'FPN':
            arch = smp.FPN
        elif conf.arch == 'Unet':
            arch = smp.Unet
        elif conf.arch == 'DeepLabV3':
            arch = smp.DeepLabV3
        else:
            assert 0, f'Unknown architecture {conf.arch}'

        weights = 'imagenet' if conf.pretrained else None
        self.model = arch(
            encoder_name=conf.backbone, encoder_weights=weights, in_channels=5,
            classes=num_classes, activation=None)

    def forward(self, x):
        x = self.model(x)
        return  x
