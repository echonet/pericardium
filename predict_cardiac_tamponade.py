import sys
import numpy as np
import pandas as pd
import os
import math
import glob
import itertools
from tqdm import tqdm
import cv2
import argparse

import torch
from lightning_utilities.core.imports import compare_version
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18

from pathlib import Path
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, precision_recall_curve

from utils import sensivity_specifity_cutoff, EchoDataset,get_frame_count, sigmoid

def process_manifest(manifest,
                     subsample: float = None,
                     limit_columns = ['filename', 'split']):
    manifest = manifest[manifest['split'] == 'test'].reset_index(drop=True)
    manifest = manifest.sample(frac = subsample) if subsample else manifest
    manifest = manifest[limit_columns] if limit_columns else manifest
    return manifest

