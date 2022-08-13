import torch
import random
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self):
        self.sample_size = 10000
        w_1 = 23
        w_2 = 43
        bias = 21

        self.x_1 = []
        self.x_2 = []
        self.y = []
        for i in range(self.sample_size):
            x_1_tmp = random.random()
            x_2_tmp = random.random()
            eps = random.random()
            y_tmp = x_1_tmp * w_1 + x_2_tmp * w_2 + bias + eps 

            self.x_1.append(x_1_tmp)
            self.x_2.append(x_2_tmp)
            self.y.append(y_tmp)
        self.to_tensor()
            
    def __len__(self):
        return self.sample_size

    def to_tensor(self):
        self.x = torch.tensor(list(zip(self.x_1, self.x_2)), requires_grad=True)
        self.y = torch.tensor(self.y, requires_grad=True)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]