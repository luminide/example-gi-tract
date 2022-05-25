#!/bin/bash -x
#
# resume training from a saved model
#

/usr/bin/time -f "Time taken: %E" python3 ../code/train.py --epochs 60 --resume model.pth
