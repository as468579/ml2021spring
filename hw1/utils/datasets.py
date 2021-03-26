import os
import numpy as np
import csv
import torch
from torch.utils.data import Dataset


class COVID19Dataset(Dataset):
    '''
        Dataset for loading and preprocessing the COVID19 dataset
    '''

    def __init__(self,
                 csv_path,
                 mode='train',
                 target_only=False):

        self.mode = mode
        
        # Read data into numpy arrays
        with open(csv_path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indeices = 57 & 75) 
            feats = list(range(40)) + [57, 75]

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]

            # Convert data into Pytorch tensors
            self.data = torch.FlaotTensor(data)

        else:
            # Training data
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day3 (18))
            target = data[:, -1]
            data   = data[:, feats]

            # Spliting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if 1 % 10 == 0]

            # Convert data into Pytorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]
        print(f'Finished reading the {self.mode} set of COVID19 Dataset ({len(data)} samples found, each dim = {self.dim})')


    def __getitem__(self, index):
        
        # Return one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]

        else:
            # FOr testing (no target)
            return self.data[index]

    def __len__(self):
        
        # Return the size of the dataset
        return len(self.data)

if __name__ == '__main__':
    train_csv = '../covid.train.csv'
    dataset = COVID19Dataset(train_csv, 'train', target_only=False)
    print(dataset[0][0].shape)
