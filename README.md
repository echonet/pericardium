# EchoNet-Effusion : XXXXXX


XXXXX
XXXXX
XXXXX
XXXXX
XXXXX
XXXXX

**Presentation:** XXXXXX

**Preprint:** Chiu IM, Vukadinovic M, Sahashi Y, Cheng PP, Cheng CY, Cheng S, Ouyang D. Automated Evaluation for Pericardial Effusion and Cardiac Tamponade with Echocardiographic Artificial Intelligence. medRxiv. 2024 Dec 1:2024.11.27.24318110. doi: 10.1101/2024.11.27.24318110. PMID: 39649606

**Paper:** XXXXX 

### Prerequisites

1. Python: we used 3.10.12
2. PyTorch we used pytorch==2.2.0
3. Other dependencies listed in `requirements.txt`

### Installation
First, clone the repository and install the required packages:


## Quickstart for inference

```sh
mkdir pericardium
cd pericardium 
git clone https://github.com/echonet/pericardium.git
pip install -r requirements.txt
```


We used [R2plus1D model](https://arxiv.org/abs/1711.11248) for echocadriography video training and inference. 
In R2+1D model, the architecture decomposes all 3D convolutions into 2D spatial convolutions followed by temporal convolutions to incorporate both spatial as well as temporal information while minimizing model size.

All you need to prepare is 
- Echocardiography Dataset (We used 112*112 AVI video, A4C/A2C/PLAX/PSAX/Subcostal echocardiography views) 
(Note: our datasets were de-identified and electrocardiogram and respirometer tracings were masked.)

- Disease LABEL file (csv) that contain `Study_Unique_ID`, `Video_Unique_ID`, `frames`, `LABEL` (LABEL will be  0/1 in binarized outcome and numerical value if regression task)

We released model weights (`/pretrained_models`) and inference code for binarized outcomes (pericardial effusion and cardiac tamponade).

Please note that for cardiac tamponade, we used A4C video for training and inference.

```sh
python predict.py  --dataset YOUR_112*112_EchoDataset  --label ['pericardial_effusion'] --view ['A4C']
```

Following running this code, you will get `Prediction_<VIEW>_<LABEL>.csv`. 
This prediction.csv has the outputs (preds) and filename and labels (From Disease LABEL file that you have). 
We concatenated all outputs for pericardial effusion predicted values. The weights (coefficient) of each view that was used in the paper were released in `coeffieient_information.csv`

Fin.