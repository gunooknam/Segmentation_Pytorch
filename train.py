import sys
import torch.nn.functional as F
import argparse

import models
parser = argparse.ArgumentParser()


parser.add_argument('--base', default='unet')
parser.add_argument('--pretrained', default="")
parser.add_argument('--data', default='data')
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--save_dir', default=None)
args = parser.parse_args()

