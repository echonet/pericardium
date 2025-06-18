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
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve

from utils import sensivity_specifity_cutoff, EchoDataset,get_frame_count, sigmoid

def process_manifest(manifest):
    manifest = manifest[manifest['split'] == 'test'].reset_index(drop=True)
    manifest = manifest[['filename', 'split']]
    return manifest

with torch.no_grad():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Predict script for Liver disease Prediction From Echocardiography.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--view", type=str, required=True, choices = ['A4C', 'A2C', 'PLAX', 'PSAX', 'SC'] ,help="Echo dataset View")
    
    args = parser.parse_args()   
    
    weights_path = f"/workspace/imin/Pericardium_Effusion_Public_Repo/pretrained_models/{args.view.lower()}_model_effusion.pt"
    
    data_path = args.dataset    #update the manifest file when needed
    all_video_files_generator = glob.iglob(os.path.join(data_path, "*.avi"))
    video_files = list(itertools.islice(all_video_files_generator, 1000))
    
    #Make manifest file from video files directory
    manifest = pd.DataFrame({"filename": video_files})
    manifest["split"] = "test"
    manifest = process_manifest(manifest)
    
    manifest["file_uid"] = manifest["filename"].apply(lambda x: os.path.basename(x))
    manifest['frames']=  manifest["filename"].apply(lambda x: get_frame_count(os.path.join(args.dataset, f"{x}")))
    manifest = manifest[manifest['frames'] > 31].reset_index(drop=True)
    
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.csv")
    print(f"Manifest file was updated and saved to {manifest_path}")
    manifest.to_csv(manifest_path, index = False)
    
    #--------------------------------------------------
    #Step: Opportunistic Liver Disease Screening
    print('--- Step: Pericardial Effusion Disease Prediction ---')
    print("Prediction Dataset view: ", args.view)
    print("Prediction Dataset path: ", data_path)
    if 'filename' and 'file_uid' and 'split' and 'frames' not in manifest.columns:
        raise ValueError("Manifest file must contain 'filename' and 'split' etc columns.")

    bs = 10
    test_ds = EchoDataset(
        split="test",
        data_path=data_path,
        manifest_path=manifest_path,
        resize_res = (112, 112),
        random_start = True
        )
    
    test_dl = DataLoader(
        test_ds, 
        num_workers=8,  
        batch_size=bs, 
        drop_last=False, 
        shuffle=False,
        )
    
    #Model loading
    pretrained_weights = torch.load(weights_path)
    new_state_dict = {}
    for k, v in pretrained_weights.items():
        new_key = k[2:] if k.startswith('m.') else k
        new_state_dict[new_key] = v
        
    backbone = r2plus1d_18(num_classes=5)
    backbone.load_state_dict(new_state_dict, strict=True)
    backbone = backbone.to(device).eval()

    #Predicting Pericardial Effusion Dataset
    #--------------------------------------------------
    
    filenames = []
    predictions_0 = [] #Pericardial Effusion None
    predictions_1 = [] #Pericardial Effusion Trivial
    predictions_2 = [] #Pericardial Effusion Small
    predictions_3 = [] #Pericardial Effusion Moderate
    predictions_4 = [] #Pericardial Effusion Severe
    
    
    for batch in tqdm(test_dl):
        batch_preds = backbone(batch["primary_input"].to(device))
        batch_preds = batch_preds.detach().cpu().squeeze(dim = 1)
        batch_filenames = batch["filename"]
        
        filenames.extend(batch["filename"])
        predictions_0.extend(batch_preds[:, 0].tolist())  
        predictions_1.extend(batch_preds[:, 1].tolist()) 
        predictions_2.extend(batch_preds[:, 2].tolist()) 
        predictions_3.extend(batch_preds[:, 3].tolist()) 
        predictions_4.extend(batch_preds[:, 4].tolist())
        
    #numpy and sigmoid
    predictions_0 = sigmoid(np.array(predictions_0))
    predictions_1 = sigmoid(np.array(predictions_1))
    predictions_2 = sigmoid(np.array(predictions_2))
    predictions_3 = sigmoid(np.array(predictions_3))
    predictions_4 = sigmoid(np.array(predictions_4))

    df_preds = pd.DataFrame({'filename': filenames, 
                             'preds_0': predictions_0,
                            'preds_1': predictions_1,
                            'preds_2': predictions_2,
                            'preds_3': predictions_3,
                            'preds_4': predictions_4}
                            
                            )
    manifest_with_preds = manifest.merge(df_preds, on="filename", how="inner").drop_duplicates('filename')
    
    manifest_with_preds.to_csv(
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "Output_Data"
        /
        Path(f"Effusion_detection_{args.view}.csv"),
        index=False,
    )
    
    print(f"Predict Pericardial effusion DETECTION -View {args.view}- was done. See Output csv and Calculate AUC")
    
#SAMPLE SCRIPT
#python predict_pericardial_effusion.py  --dataset "/workspace/data/drives/sdb/pericardial_effusion_echo_video/"  --view A4C