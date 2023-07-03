import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_data
from torchtext


class imageCaptionsDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = get_data(data_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass


