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


class MultiTaskDataloader(object):
    def __init__(self, tau=1.0, **dataloaders):
        self.dataloaders = dataloaders
        Z = sum(pow(v, tau) for v in self.dataloader_sizes.values())
        self.tasknames, self.sampling_weights = zip(*((k, pow(v, tau) / Z) for k, v in self.dataloader_sizes.items()))
        self.dataiters = {k: self.cycle(v) for k, v in dataloaders.items()}

    @property
    def dataloader_sizes(self):
        if not hasattr(self, '_dataloader_sizes'):
            self._dataloader_sizes = {k: len(v) for k, v in self.dataloaders.items()}

        return self._dataloader_sizes
        
    def cycle(self,iterable):
        while True:
            for x in iterable:
                yield x

    def __len__(self):
        return sum(v for k, v in self.dataloader_sizes.items())

    def __iter__(self):
        for i in range(len(self)):
            taskname = np.random.choice(self.tasknames, p=self.sampling_weights)
            dataiter = self.dataiters[taskname]
            batch = next(dataiter)
            batch['task'] = taskname
            yield batch


# For both UDPOS and PANX:
class TokenClassifyDataset(Dataset):
    def __init__(self, task_tuple: Tuple[List[str], List[int], List[str]], task):
        self.texts = task_tuple[0]
        self.labels = task_tuple[1]
        self.unique_labels = task_tuple[2]
        self.task = task

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[int, torch.tensor]:

        item = dict()
        item['text'] = self.texts[idx]
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.int64)
        item['task'] = self.task
        return item

class XNLIDataset(Dataset):
    def __init__(self, task_tuple: Tuple[pd.DataFrame, List[str]]):
        self.text1 = task_tuple[0].iloc[:, 0].tolist()
        self.text2 = task_tuple[0].iloc[:, 1].tolist()
        self.labels = task_tuple[0].iloc[:, 2].tolist()
        self.unique_labels = task_tuple[1]

    def __len__(self) -> int:
        return len(self.text1)

    def __getitem__(self, idx: int) -> Dict[int, torch.tensor]:

        item = dict()
        item['text1'] = self.text1[idx]
        item['text2'] = self.text2[idx]
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.int64)
        item['task'] = 'xnli'
        return item

class PAWSXDataset(Dataset):
    def __init__(self, task_df: pd.DataFrame):
        self.text1 = task_df.iloc[:, 0].tolist()
        self.text2 = task_df.iloc[:, 1].tolist()
        self.labels = task_df.iloc[:, 2].tolist()

    def __len__(self) -> int:
        return len(self.text1)

    def __getitem__(self, idx: int) -> Dict[int, torch.tensor]:

        item = dict()
        item['text1'] = self.text1[idx]
        item['text2'] = self.text2[idx]
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.int64)
        item['task'] = 'pawsx'
        return item


# mMiniLM: distilled version of XLM-R model
class mMiniLM(pl.LightningModule):
    def __init__(self, model_hparams):
        super(mMiniLM, self).__init__()

        self.model_hparams = model_hparams
        self.mMiniLM = AutoModel.from_pretrained(self.model_hparams['base_model'])

        self.classifier_udpos = AutoModelForTokenClassification.from_pretrained(self.model_hparams['base_model'], num_labels=17).classifier
        self.classifier_panx = AutoModelForTokenClassification.from_pretrained(self.model_hparams['base_model'], num_labels=7).classifier
        freeze_base_model = False
        self.classifier_pawsx = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(384, 2),
            nn.Softmax(dim=-1)
        )

        self.classifier_xnli = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(384, 3),
            nn.Softmax(dim=-1)
        )

        # Freeze base model layers and only train the classification layer weights
        if freeze_base_model:
            for p in self.mMiniLM.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, task) -> torch.tensor:

        features = self.mMiniLM(input_ids, attention_mask)

        if task == "panx":
            out = self.classifier_panx(features[0])

        elif task == "pawsx":
            out = self.classifier_pawsx(features[1])

        elif task == "udpos":
            out = self.classifier_udpos(features[0])

        elif task == "xnli":
            out = self.classifier_xnli(features[1])

        return out


    def training_step(self, batch: torch.tensor, batch_idx: int) -> torch.tensor:

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        labels = batch['labels']
        task = batch['task']
        self.task = batch['task']

        # Forward pass
        outputs = self(input_ids, attention_mask, task)

        CEL = nn.CrossEntropyLoss(ignore_index=-100)

        if task == "panx" or task == "udpos":
            loss = torch.tensor(0)
            for i, o in enumerate(outputs):
                loss = loss.add(CEL(o, labels[i]))
            loss = torch.div(loss, outputs.shape[0])

        else:
            loss = CEL(outputs, labels)

        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False,):

        optimizer_closure()
        
        if self.task == 'udpos' or self.task == "panx":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.model_hparams['token_learning_rate'])
        elif self.task == 'pawsx' or self.task == "xnli":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.model_hparams['sentence_learning_rate'])

        optimizer.step()

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> Dict:

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        labels = batch['labels']
        task=batch['task']

        # Forward pass
        outputs = self(input_ids, attention_mask, task)

        CEL = nn.CrossEntropyLoss(ignore_index=-100)

        if task == "panx" or task == "udpos":

            val_acc = -1.0
            batch_word_ids = batch['word_ids']
            loss = torch.tensor(0)
            y = []
            l = []

            for i, o in enumerate(outputs):
                loss = loss.add(CEL(o, labels[i]))
                y_pred = torch.max(o, -1)[1] # compute highest-scoring tag (fetch only argmax tensor from output tuple)
                
                word_ids = batch_word_ids[i].tolist()
                unique_word_ids = set(word_ids) # taking the unique word_ids
                unique_word_ids.remove(-100) # removing the -100 word_id which was used in place of None values

                for u_wid in unique_word_ids:
                    orig_id = word_ids.index(u_wid)
                    if labels[i][orig_id] != -100:
                        y.append(y_pred[orig_id])
                        l.append(labels[i][orig_id])                        
            loss = torch.div(loss, outputs.shape[0])

            if y:    
                num_classes = 17 if task == "udpos" else 7
                f1 = torchmetrics.F1(num_classes=num_classes).to(self.device)
                f1_score = f1(torch.tensor(y), torch.tensor(l)).item()
            else:
                fl_score = -1.0

        else:

            f1_score = -1.0
            loss = CEL(outputs, labels)
            accuracy = torchmetrics.Accuracy().to(self.device)
            val_acc = accuracy(torch.argmax(outputs, dim=1), labels).item()
                
        return {"val_loss": loss, 'val_acc': val_acc, "val_f1": f1_score}

    def validation_epoch_end(self, outputs: List) -> Dict:
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = np.mean([x['val_acc'] for x in outputs if x['val_acc'] >= 0])
        avg_val_f1 = np.mean([x['val_f1'] for x in outputs if x['val_f1'] >= 0])

        self.log("val_loss", avg_loss)
        self.log("val_acc", avg_val_acc, prog_bar=True)
        self.log("val_f1", avg_val_f1, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Adam:

        return torch.optim.Adam(self.parameters(), lr=self.model_hparams['token_learning_rate'])

    def __setstate__(self, d):
        self.__dict__ = d
        self.cuda()
        self.eval()

    def set_labels(self, T):
        self.udpos_unique_labels, self.panx_unique_labels, self.xnli_unique_labels = T


    def predict_udpos(self, filename: str) -> List[List[str]]:
        with torch.no_grad():
            predictions = []
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            udpos_mapping = self.udpos_unique_labels
            tokenizer = AutoTokenizer.from_pretrained(self.model_hparams['base_model'])
            sentence = []
            sentences = []
            with open(filename, 'r') as f:
                data = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
                if data[-1]:
                    data.append([])
            for line in data:
                if line:
                    sentence.append(line[0])
                else:
                    sentences.append(sentence)
                    sentence = []
            
            batch_size = 64
            sentences = np.array([[value, index] for index, value in sorted(enumerate(sentences), key=lambda x: len(x[1]))], dtype=object)
            sentences, indices = sentences[:, 0].tolist(), sentences[:, 1].tolist()
            batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

            for batch in batches:
                encodings_orig = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors='pt', is_split_into_words=True)
                encodings = {key: torch.tensor(val).to(device) for key,val in encodings_orig.items()}
                logits = self(encodings['input_ids'], encodings['attention_mask'], "udpos")
                logits = logits.to('cpu')
                for k,sent in enumerate(logits):
                    word_ids = encodings_orig.word_ids(batch_index=k)
                    label = []
                    for logit in sent: 
                        label.append(udpos_mapping[(np.argmax(logit.numpy()))])
                    correct_label = [label[word_ids.index(i)] if i in word_ids else 'SYM' for i in range(len(sent))]
                    correct_label = correct_label[:len(batch[k])]
                    predictions.append(correct_label)
            predictions = [predictions[indices.index(idx)] for idx in range(len(predictions))]
        return predictions

    def predict_panx(self, filename: str) -> List[List[str]]:
        with torch.no_grad():
            predictions = []
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            panx_mapping = self.panx_unique_labels
            tokenizer = AutoTokenizer.from_pretrained(self.model_hparams['base_model'])
            sentence = []
            sentences = []
            with open(filename, 'r') as f:
                data = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
                if data[-1]:
                    data.append([])
            for line in data:
                if line:
                    sentence.append(line[0])
                else:
                    sentences.append(sentence)
                    sentence = []

            batch_size = 64
            sentences = np.array([[value, index] for index, value in sorted(enumerate(sentences), key=lambda x: len(x[1]))], dtype=object)
            sentences, indices = sentences[:, 0].tolist(), sentences[:, 1].tolist()
            batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

            for batch in batches:
                encodings_orig = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors='pt', is_split_into_words=True)
                encodings = {key: torch.tensor(val).to(device) for key,val in encodings_orig.items()}
                logits = self(encodings['input_ids'], encodings['attention_mask'], "panx")
                logits = logits.to('cpu')
                for k,sent in enumerate(logits):
                    word_ids = encodings_orig.word_ids(batch_index=k)
                    label = []
                    for logit in sent: 
                        label.append(panx_mapping[(np.argmax(logit.numpy()))])
                    correct_label = [label[word_ids.index(i)] if i in word_ids else 'O' for i in range(len(sent))]
                    correct_label = correct_label[:len(batch[k])]
                    predictions.append(correct_label)
            predictions = [predictions[indices.index(idx)] for idx in range(len(predictions))]
        return predictions

    def predict_xnli(self, filename: str) -> List[str]:
        with torch.no_grad():
            predictions = []
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            xnli_mapping = self.xnli_unique_labels
            tokenizer = AutoTokenizer.from_pretrained(self.model_hparams['base_model'])

            df = pd.read_csv(filename, skip_blank_lines=False, header=None, sep="\t", quoting=csv.QUOTE_NONE)
            #df = pd.read_csv(filename, skip_blank_lines=False, usecols=[0,1,2], header=None, sep="\t", quoting=csv.QUOTE_NONE)
            df.dropna(inplace=True)
            xnli_text1 = df.iloc[:, 0].tolist()
            xnli_text2 = df.iloc[:, 1].tolist()
            
            batch_size = 64
            xnli_text1 = np.array([[value1, value2, index] for index, (value1, value2) in sorted(enumerate(zip(xnli_text1,xnli_text2)), key=lambda x: len(x[1][0])+len(x[1][1]))], dtype=object)
            xnli_text1, xnli_text2, indices = xnli_text1[:, 0].tolist(), xnli_text1[:, 1].tolist(), xnli_text1[:, 2].tolist()

            batches_1 = [xnli_text1[i:i + batch_size] for i in range(0, len(xnli_text1), batch_size)]
            batches_2 = [xnli_text2[i:i + batch_size] for i in range(0, len(xnli_text2), batch_size)]

            for batch_1, batch_2 in zip(batches_1, batches_2):
                encodings = tokenizer(batch_1, batch_2, truncation=True, padding=True, max_length=512, return_tensors='pt')
                encodings = {key: torch.tensor(val).to(device) for key,val in encodings.items()}
                logits = self(encodings['input_ids'], encodings['attention_mask'], "xnli")
                logits = logits.to('cpu')
                for logit in logits:
                    label = xnli_mapping[(np.argmax(logit.numpy()))]
                    predictions.append(label)
            predictions = [predictions[indices.index(idx)] for idx in range(len(predictions))]
        return predictions

    def predict_pawsx(self, filename: str) -> List[str]:
        with torch.no_grad():
            predictions = []
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #pawsx_mapping = self.pawsx_unique_labels
            tokenizer = AutoTokenizer.from_pretrained(self.model_hparams['base_model'])
            df = pd.read_csv(filename, skip_blank_lines=False, header=None, sep="\t", quoting=csv.QUOTE_NONE)
            df.dropna(inplace=True)
            pawsx_text1 = df.iloc[:, 0].tolist()
            pawsx_text2 = df.iloc[:, 1].tolist()

            batch_size = 64
            pawsx_text1 = np.array([[value1, value2, index] for index, (value1, value2) in sorted(enumerate(zip(pawsx_text1,pawsx_text2)), key=lambda x: len(x[1][0])+len(x[1][1]))], dtype=object)
            pawsx_text1, pawsx_text2, indices = pawsx_text1[:, 0].tolist(), pawsx_text1[:, 1].tolist(), pawsx_text1[:, 2].tolist()

            batches_1 = [pawsx_text1[i:i + batch_size] for i in range(0, len(pawsx_text1), batch_size)]
            batches_2 = [pawsx_text2[i:i + batch_size] for i in range(0, len(pawsx_text2), batch_size)]

            for batch_1, batch_2 in zip(batches_1, batches_2):
                encodings = tokenizer(batch_1, batch_2, truncation=True, padding=True, max_length=512, return_tensors='pt')
                encodings = {key: torch.tensor(val).to(device) for key,val in encodings.items()}
                logits = self(encodings['input_ids'], encodings['attention_mask'], "pawsx")
                logits = logits.to('cpu')
                for logit in logits:
                    label = np.argmax(logit.numpy())
                    predictions.append(label)
            predictions = [predictions[indices.index(idx)] for idx in range(len(predictions))]
        return predictions


# load and preprocess all task data
def load_and_preprocess_data():

    # To store all train/val data according to their task names. Each key (task name) holds a Tuple that contains task-specific data
    dict_train = dict()
    dict_val = dict()

    for task in task_list:

        if task == 'udpos':
            
            sentences = []
            sentence = []
            labels = []
            label = []
            data = []
            for file in os.listdir(os.path.join(path, task)):
                with open(os.path.join(path, task, file), 'r') as f:
                    data += list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
                    if data[-1]:
                        data.append([])

            for line in data:

                if line:
                    sentence.append(line[0])
                    label.append(line[1])

                else:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            unique_udpos_labels = list(set([l for label in labels for l in label]))
            labels = [[unique_udpos_labels.index(l) for l in label] for label in labels]

            sentences_train, sentences_val, labels_train, labels_val = train_test_split(sentences, labels, test_size=0.10, random_state=42)
            dict_train[task] = (sentences_train, labels_train, unique_udpos_labels)
            dict_val[task] = (sentences_val, labels_val, unique_udpos_labels)

        elif task == 'panx':

            sentences = []
            sentence = []
            labels = []
            label = []
            data = []
            for file in os.listdir(os.path.join(path, task)):
                with open(os.path.join(path, task, file), 'r') as f:
                    data += list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
                    if data[-1]:
                        data.append([])

            for line in data:

                if line:
                    sentence.append(line[0])
                    label.append(line[1])

                else:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            unique_panx_labels = list(set([l for label in labels for l in label]))
            labels = [[unique_panx_labels.index(l) for l in label] for label in labels]
            
            sentences_train, sentences_val, labels_train, labels_val = train_test_split(sentences, labels, test_size=0.10, random_state=42)
            dict_train[task] = (sentences_train, labels_train, unique_panx_labels)
            dict_val[task] = (sentences_val, labels_val, unique_panx_labels)

        elif task == 'xnli':
            df_chunks = [pd.read_csv(os.path.join(path, task, file), header=None, sep='\t', quoting=csv.QUOTE_NONE) for file in os.listdir(os.path.join(path, task))]
            df = pd.concat(df_chunks, ignore_index=True)  # contains data of current task
            df.dropna(inplace=True)

            # changing string labels to index (integer) based labels
            xnli_labels = df.iloc[:, 2].tolist()
            unique_xnli_labels = list(set(xnli_labels))
            replace_labels_dict = {label:i for i, label in enumerate(unique_xnli_labels)}
            df.iloc[:, 2] = df.iloc[:, 2].map(replace_labels_dict)

            df_xnli_train, df_xnli_val = train_test_split(df, test_size=0.10, random_state=42)
            dict_train[task] = (df_xnli_train, unique_xnli_labels)
            dict_val[task] = (df_xnli_val, unique_xnli_labels)

        elif task == 'pawsx':
            df_chunks = [pd.read_csv(os.path.join(path, task, file), header=None, sep='\t', quoting=csv.QUOTE_NONE) for file in os.listdir(os.path.join(path, task))]
            df = pd.concat(df_chunks, ignore_index=True)  # contains data of current task
            df.dropna(inplace=True)
            dict_train[task], dict_val[task] = train_test_split(df, test_size=0.10, random_state=42)


    return dict_train, dict_val

def tokenize_and_align_labels(text, labels):

    encodings = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt', is_split_into_words=True)
    aligned_labels = []
    aligned_word_ids = []
    label_all_tokens = False

    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        new_word_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
                new_word_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
                new_word_ids.append(word_idx)
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                new_word_ids.append(word_idx)

            previous_word_idx = word_idx

        aligned_labels.append(label_ids)
        aligned_word_ids.append(new_word_ids)

    return encodings, torch.tensor(aligned_labels, dtype=torch.int64), torch.tensor(aligned_word_ids)

def collate_fn(data):

    batch = dict()
    keys = data[0].keys()

    # converting list of dictionaries to dictionary of lists
    for k in keys:
        batch[k] = [data[i][k] for i in range(len(data))]
    
    task = batch['task'][0]
    labels = batch['labels']

    if task == "panx" or task == "udpos":
        text = batch['text']

        encodings, labels, batch_word_ids = tokenize_and_align_labels(text, labels)
        batch = {key: torch.tensor(val) for key,val in encodings.items()}
        batch['labels'] = labels
        batch['word_ids'] =  batch_word_ids
        batch['task'] = [task] * len(data)

    else:
        text1 = batch['text1']
        text2 = batch['text2']

        encodings = tokenizer(text1, text2, truncation=True, padding=True, max_length=512, return_tensors='pt')  
        batch = {key: torch.tensor(val) for key,val in encodings.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        batch['task'] = [task] * len(data)
                
    return batch


if __name__ == "__main__":

    task_list = ['panx', 'pawsx', 'udpos', 'xnli']
    path = "../../../data/"

    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # same name as model class name
    chosen_model = 'mMiniLM'
    
    # Hyper-parameters for current model
    model_hparams = cfg[chosen_model]

    dict_train, dict_val = load_and_preprocess_data()

    tokenizer = AutoTokenizer.from_pretrained(model_hparams['base_model'])

    panx_train = TokenClassifyDataset(dict_train['panx'], 'panx')
    pawsx_train = PAWSXDataset(dict_train['pawsx'])
    udpos_train = TokenClassifyDataset(dict_train['udpos'], 'udpos')
    xnli_train = XNLIDataset(dict_train['xnli'])

    panx_val = TokenClassifyDataset(dict_val['panx'], 'panx')
    pawsx_val = PAWSXDataset(dict_val['pawsx'])
    udpos_val = TokenClassifyDataset(dict_val['udpos'], 'udpos')
    xnli_val = XNLIDataset(dict_val['xnli'])

    panx_dataloader =  torch.utils.data.DataLoader(dataset=panx_train, collate_fn=collate_fn, batch_size=model_hparams['batch_size'], num_workers=4, shuffle=True)
    pawsx_dataloader =  torch.utils.data.DataLoader(dataset=pawsx_train, collate_fn=collate_fn, batch_size=model_hparams['batch_size'], num_workers=4, shuffle=True)
    udpos_dataloader =  torch.utils.data.DataLoader(dataset=udpos_train, collate_fn=collate_fn, batch_size=model_hparams['batch_size'], num_workers=4, shuffle=True)
    xnli_dataloader =  torch.utils.data.DataLoader(dataset=xnli_train, collate_fn=collate_fn, batch_size=model_hparams['batch_size'], num_workers=4, shuffle=True)
    train_loader = MultiTaskDataloader(1.0, panx = panx_dataloader, pawsx = pawsx_dataloader, udpos = udpos_dataloader, xnli = xnli_dataloader)

    panx_dataloader =  torch.utils.data.DataLoader(dataset=panx_val, collate_fn=collate_fn, batch_size=model_hparams['batch_size'], num_workers=4, shuffle=True)
    pawsx_dataloader =  torch.utils.data.DataLoader(dataset=pawsx_val, collate_fn=collate_fn, batch_size=model_hparams['batch_size'], num_workers=4, shuffle=True)
    udpos_dataloader =  torch.utils.data.DataLoader(dataset=udpos_val, collate_fn=collate_fn, batch_size=model_hparams['batch_size'], num_workers=4, shuffle=True)
    xnli_dataloader =  torch.utils.data.DataLoader(dataset=xnli_val, collate_fn=collate_fn, batch_size=model_hparams['batch_size'], num_workers=4, shuffle=True)
    test_loader = MultiTaskDataloader(1.0, panx = panx_dataloader, pawsx = pawsx_dataloader, udpos = udpos_dataloader, xnli = xnli_dataloader)


    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./model_weights/',
        filename='mMiniLM-finetuned-{epoch:02d}-{val_acc:.2f}',
        mode="min",
    )

    model = globals()[chosen_model](model_hparams)

    trainer = Trainer(gpus=-1, auto_select_gpus=True, max_epochs=model_hparams['num_epochs'], callbacks=[checkpoint_callback])
    # trainer = Trainer(max_epochs=num_epochs, callbacks=[checkpoint_callback])   # To test run in the local machine
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    # model = model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    model.eval()

    state = {
        'labels': (udpos_train.unique_labels, panx_train.unique_labels, xnli_train.unique_labels),
        'state_dict': model.state_dict()
    }

    with open("../../state_mMiniLM_L12xH384.p3", "wb") as f:
        pickle.dump(state, f)
