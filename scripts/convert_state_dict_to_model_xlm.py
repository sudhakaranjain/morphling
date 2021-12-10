from typing import Dict, List, Tuple
import csv
import os
import pickle

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torchinfo import summary
from torchvision.transforms.functional import normalize, to_tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoModel, AutoTokenizer, BertTokenizer, BertModel, pipeline
from tqdm import tqdm
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import yaml
#from morphling.mbert_mtldataloader import mBertBaseline
#from morphling.mbert_mtldataloader import mBertBaseline
from morphling.mtl_model import mMiniLM
#from morphling.mbert_mtldataloader_experimental import mBertBaseline

with open("../morphling/config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# same name as model class name
chosen_model = 'mMiniLM'

# Hyper-parameters for current model
model_hparams = cfg[chosen_model]

model = globals()[chosen_model](model_hparams)
with open("../../state_mMiniLM_L12xH384.p3", "rb") as f:
    state = pickle.load(f)

model.load_state_dict(state_dict=state['state_dict'])
model.set_labels(state['labels'])
model.eval()

with open("../../model_mMiniLM_L12xH384.p3", "wb") as f:
    pickle.dump(model, f)
