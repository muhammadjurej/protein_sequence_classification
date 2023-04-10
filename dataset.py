import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import tools
import os
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class DatasetProtein(Dataset):
    def __init__(self, root, split, amino, common_family, encoding="integer", total_sample_size=541, max_size=250, device='cuda'):
        self.root = root
        self.split = split
        self.device = device
        self.max_size = max_size
        self.total_sample_size = total_sample_size
        self.save_path = os.path.join("data/", self.split + 'no_categorical_data.pth')
        self.tool = tools.Tools()
        self.amino = amino #lis of common amino acids
        self.data_encode = encoding
        self.common_family=common_family

        if os.path.exists(self.save_path):
            print(f'Loading dataset {self.split} ...')
            self.filtered_data = self.filter_data()
            pth_dataset = torch.load(self.save_path)
            self.data, self.labels = pth_dataset['data'].to(self.device), pth_dataset['label'].to(self.device)
            print(f"Dataset {self.split} loaded !")
        else:
            print(f"File {self.save_path} does not exist. Creating dataset...")  
            self.data, self.labels = self.get_dataset()
            self.data, self.labels = torch.from_numpy(self.data), torch.from_numpy(self.labels)
            data_dict = {"data": self.data, "label": self.labels}
            self.data, self.labels = self.data.to(self.device), self.labels.to(self.device)  
            print(f"Dataset {self.split} created !")
            torch.save(data_dict, self.save_path)
            print(f"File {self.save_path} Saved!")

    def load(self):
        self.data = []
        for fn in os.listdir(os.path.join(self.root, self.split)):
            with open(os.path.join(self.root, self.split, fn)) as f:
                self.data.append(pd.read_csv(f, index_col=None))
        return pd.concat(self.data)

    #filter for most common family on dataset
    def filter_data(self):
        self.df = self.load()
        self.mask = self.df.family_accession.isin(self.common_family.index.values)

        if self.split == 'train':
            self.filter_df = self.df.loc[self.mask,:]
            self.nrows = len(self.filter_df)
            self.filter_df.groupby('family_accession').head()
            self.filter_df = self.filter_df.groupby('family_accession', group_keys=False).apply(lambda x: x.sample(self.total_sample_size))
            self.filter_df = self.filter_df.sample(frac=1).reset_index(drop=True)
        else:
            self.filter_df = self.df.loc[self.mask, :]
        return self.filter_df

    def get_dataset(self):
        encode = LabelEncoder()
        self.filtered_data = self.filter_data()
        self.amino_dict = self.tool.create_dict(self.amino)
        
        if(self.data_encode=='integer'):
            self.encode_list = []
            for row in self.filtered_data['sequence'].values:
                self.row_encode = []
                for code in row:
                    self.row_encode.append(self.amino_dict.get(code, 0))
                self.encode_list.append(np.array(self.row_encode))

            self.data_encoded = pad_sequences(self.encode_list, maxlen=self.max_size, padding='post', truncating='post')
            if self.split=='train':
                self.label_encoded = encode.fit_transform(self.filtered_data['family_accession'])
            else:
                self.label_encoded = encode.fit_transform(self.filtered_data['family_accession'])           
        return self.data_encoded, self.label_encoded

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.filtered_data)