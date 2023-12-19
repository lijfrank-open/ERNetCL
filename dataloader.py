import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np

class IEMOCAPRobertaDataset(Dataset):

    def __init__(self, path, split):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open(path, 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in np.array(self.speakers[vid])]),\
               torch.FloatTensor([1]*len(np.array(self.labels[vid]))),\
               torch.LongTensor(np.array(self.labels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<5 else pad_sequence(dat[i]) if i<7 else dat[i].tolist() for i in dat]
    

class MELDRobertaDataset(Dataset):

    def __init__(self, path, split, classify='emotion'):
        
        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open(path, 'rb'), encoding='latin1')
        
        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.FloatTensor(np.array(self.speakers[vid])),\
               torch.FloatTensor([1]*len(np.array(self.labels[vid]))),\
               torch.LongTensor(np.array(self.labels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<5 else pad_sequence(dat[i]) if i<7 else dat[i].tolist() for i in dat]


class DailyDialogueRobertaDataset(Dataset):

    def __init__(self, path, split):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open(path, 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in np.array(self.speakers[vid])]),\
               torch.FloatTensor([1]*len(np.array(self.labels[vid]))),\
               torch.LongTensor(np.array(self.labels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<5 else pad_sequence(dat[i]) if i<7 else dat[i].tolist() for i in dat]

class EmoryNLPRobertaDataset(Dataset):

    def __init__(self, path, split, classify='emotion'):
        
        self.speakers, self.emotion_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainId, self.testId, self.validId \
        = pickle.load(open(path, 'rb'), encoding='latin1')
        
        sentiment_labels = {}
        for item in self.emotion_labels:
            array = []
            for e in self.emotion_labels[item]:
                if e in [1, 4, 6]:
                    array.append(0)
                elif e == 3:
                    array.append(1)
                elif e in [0, 2, 5]:
                    array.append(2)
            sentiment_labels[item] = array
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]
            
        if classify == 'emotion':
            self.labels = self.emotion_labels
        elif classify == 'sentiment':
            self.labels = sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(np.array(self.labels[vid]))),\
               torch.LongTensor(np.array(self.labels[vid])),\
               vid

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<5 else pad_sequence(dat[i]) if i<7 else dat[i].tolist() for i in dat]
