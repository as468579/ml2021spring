
import torch 
from torch.utils.data import Dataset

import numpy as np

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):

        # Convert from float64 to float32
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)

        else:
            self.label = None

    def __getitem__(self, index):

        if self.label is not None:
            return self.data[index], self.label[index]

        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)