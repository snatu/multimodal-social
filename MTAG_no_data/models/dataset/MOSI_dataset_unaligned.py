import os
import pickle
import numpy as np
import torch
import torch.utils.data as Data
from consts import GlobalConsts as gc

class MultimodalSubdata():
    def __init__(self, name="train"):
        self.name = name
        self.text = np.empty(0)
        self.audio = np.empty(0)
        self.vision = np.empty(0)
        self.y = np.empty(0)
        self.id = None

class MosiDatasetUnaligned(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, clas="train", for_vis=False):
        self.root = root
        self.clas = clas
        if len(MosiDatasetUnaligned.trainset.y) != 0 and clas != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.clas == "train":
            self.dataset = MosiDatasetUnaligned.trainset
        elif self.clas == "test":
            self.dataset = MosiDatasetUnaligned.testset
        elif self.clas == "valid":
            self.dataset = MosiDatasetUnaligned.validset

        self.text = self.dataset.text
        self.audio = self.dataset.audio
        self.vision = self.dataset.vision
        self.y = self.dataset.y
        self.ids = self.dataset.id

        self.for_vis = for_vis

    def load_data(self):
        if gc['data_path'][-1] != '/':
            gc['data_path'] = gc['data_path'] + '/'
        dataset = pickle.load(open(gc['data_path'] + 'mosi_data_noalign.pkl', 'rb'))
        gc['padding_len'] = dataset['test']['text'].shape[1]
        gc['text_dim'] = dataset['test']['text'].shape[2]
        gc['audio_dim'] = dataset['test']['audio'].shape[2]
        gc['vision_dim'] = dataset['test']['vision'].shape[2]

        for ds, split_type in [(MosiDatasetUnaligned.trainset, 'train'), (MosiDatasetUnaligned.validset, 'valid'),
                               (MosiDatasetUnaligned.testset, 'test')]:
            ds.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
            ds.audio = torch.tensor(dataset[split_type]['audio'].astype(np.float32))
            ds.audio[ds.audio == -np.inf] = 0
            ds.audio = ds.audio.clone().cpu().detach()
            ds.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
            ds.y = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
            ds.id = dataset[split_type]['id'].astype(str)[:, 0]

    def __getitem__(self, index):
        inputLen = len(self.text[index])
        if self.for_vis:
            return self.text[index], self.audio[index], self.vision[index], \
                   inputLen, self.y[index].squeeze(), [self.ids[index]]
        return self.text[index], self.audio[index], self.vision[index], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MosiDatasetUnaligned(gc['data_path'])