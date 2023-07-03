import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from get_loader import imageCaptionsDataset
from models import *


def train(batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    dataset = imageCaptionsDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=)
    enc_img = encodeImage()






if __name__ == '__main__':
    pass