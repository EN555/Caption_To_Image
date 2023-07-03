import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_data
from transformers import AutoTokenizer


class imageCaptionsDataset(Dataset):
    def __init__(self, tokenizer):
        self.data = get_data()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        caption = self.tokenizer.encode(self.data[idx][1])
        return image, caption


class MyCollate:
    def __init__(self, pad_idx):
        pass


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    print(imageCaptionsDataset(tokenizer)[1])