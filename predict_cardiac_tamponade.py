import sys
import numpy as np
import pandas as pd
import os
import math
import glob
import itertools
from tqdm import tqdm
tqdm.pandas()
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



with torch.no_grad():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Predict script for Liver disease Prediction From Echocardiography.")
    parser.add_argument("--dataset", type=str, required = True, help="Path to the dataset directory.")
    parser.add_argument("--view", type=str, required = True, choices = ['A4C'] ,help="Echo dataset View, for tamponade we used A4C view.")
    parser.add_argument("--preset_manifest_path", type=str, required=False,help="preset_manifest_path (A4C) which include video_data_path and split and labels")
    
    args = parser.parse_args()   
    data_path = args.dataset
    
    if args.preset_manifest_path:
        print("\nHaving preset manifest path, using it to load the manifest file.")
        print("if you want to calcluate AUC, the manifest file must contain 'pe_grade' and each grade columns.")
        manifest = pd.read_csv(args.preset_manifest_path)
        if 'split' not in manifest.columns:
            raise ValueError("Manifest file must contain 'split' columns.")
        
        manifest = process_manifest(manifest,
                                    subsample= 0.9,
                                    limit_columns= ["file_uid", "split", "pe_grade"])
        
        if not manifest["file_uid"].str.endswith('.avi').all():
            print("Adding .avi extension to file_uid in the manifest.")
            manifest["file_uid"] = manifest["file_uid"] + ".avi"
            #file_uid is like "video_1.avi", "video_2.avi" etc.
        
        if 'filename' not in manifest.columns:
            print("Creating filename from file_uid and data_path.")
            manifest["filename"] = manifest["file_uid"].apply(lambda x: os.path.join(data_path, f"{x}"))
            #filename is like "/path/to/dataset/video_1.avi", "/path/to/dataset/video_2.avi" etc.
        
        manifest['frames'] = manifest["filename"].progress_apply(lambda x: get_frame_count(os.path.join(args.dataset, f"{x}")))
        manifest = manifest[manifest['frames'] > 31].reset_index(drop=True)
        
        manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest_tmp.csv")
        manifest.to_csv(manifest_path, index = False)
    
    elif not args.preset_manifest_path:
        print("No preset manifest path provided, creating a new manifest file from the dataset directory.")
        DEBUG_N = 100 # if you want to use all, set it to None or a large/infinite number
        #update the manifest file when needed, make generator for all video files
        all_video_files_generator = glob.iglob(os.path.join(data_path, "*.avi"))
        video_files = list(itertools.islice(all_video_files_generator, DEBUG_N))
        
        #Make manifest file from video files directory
        manifest = pd.DataFrame({"filename": video_files})
        manifest["split"] = "test"
        manifest = process_manifest(manifest,
                                    # subsample= 0.1,
                                    limit_columns= None)
        
        manifest["file_uid"] = manifest["filename"].apply(lambda x: os.path.basename(x))
        manifest['frames']=  manifest["filename"].apply(lambda x: get_frame_count(os.path.join(args.dataset, f"{x}")))
        manifest = manifest[manifest['frames'] > 31].reset_index(drop=True)
        
        manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest_tmp.csv")
        print(f"Manifest file was updated and saved to {manifest_path}")
        manifest.to_csv(manifest_path, index = False)
    
    #--------------------------------------------------
    print('--- Step: Cardiac Tamponade Disease Prediction ---')
    print("Prediction Dataset view (for tamponade, view should be A4C): ", args.view)
    if args.view.lower() not in ['a4c']:
        raise ValueError("For Pericardial Effusion detection, the view should be A4C.")
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
    weights_path = f"./pretrained_models/a4c_tamponade_run-20241013_133532-tamponade_a4c_20241013_2zw_max_epoch_50_earlystopping_5_weights_model_best_epoch_val_mean_roc_auc.pt"
    pretrained_weights = torch.load(weights_path)
    new_state_dict = {}
    for k, v in pretrained_weights.items():
        new_key = k[2:] if k.startswith('m.') else k
        new_state_dict[new_key] = v
        
    backbone = r2plus1d_18(num_classes=1)
    backbone.load_state_dict(new_state_dict, strict=True)
    backbone = backbone.to(device).eval()

    #Predicting Pericardial Effusion Dataset
    #--------------------------------------------------
    
    filenames = []
    predictions = []
    
    for batch in tqdm(test_dl):
        batch_preds = backbone(batch["primary_input"].to(device))
        batch_preds = batch_preds.detach().cpu().squeeze(dim = 1)
        batch_filenames = batch["filename"]
        filenames.extend(batch["filename"])
        predictions.extend(batch_preds.tolist())
        
    # Numpy and sigmoid
    predictions = sigmoid(np.array(predictions))


    df_preds = pd.DataFrame({'filename': filenames, 
                             'preds': predictions}
                            
                            )
    manifest_with_preds = manifest.merge(df_preds, on="filename", how="inner").drop_duplicates('filename')
    Output_path = Path(os.path.dirname(os.path.abspath(__file__))) / "Output_Data" /Path(f"Effusion_detection_{args.view}.csv")
    manifest_with_preds.to_csv(
        Output_path,
        index=False,
    )
    
    print(f"Predict Tamponade DETECTION -View {args.view}- was done. See {Output_path} and Calculate AUC")
    
    # if "pe_grade" in manifest.columns:
    #     # Calculate AUC for each class
    #     # None (0), Trivial (1), Small (2), Moderate (3), Severe (4)
        
    #     #None 
    #     y_turue_0 = manifest['pe_grade_none'].values
    #     y_pred_0 = manifest_with_preds['preds_0'].values
    #     auc_0 = roc_auc_score(y_turue_0, y_pred_0)
    #     print(f"AUC for Pericardial Effusion None: {auc_0:.3f}")

    # else:
    #     print("No 'pe_grade' column in the manifest file. AUC calculation is skipped.")
    
    #MEMO TODO CHECK
    /workspace/imin/echo_tamponade/test_tamponade.ipynb
    CHECK and how to calculate all AUC
        
    
# SAMPLE SCRIPT
# python predict_cardiac_tamponade.py  --dataset "/workspace/data/drives/sdb/pericardial_effusion_echo_video/" --view "A4C" 
# --preset_manifest_path  /workspace/imin/echo_tamponade_manifest/manifest_video_pe_tamponade_class_a4c.csv