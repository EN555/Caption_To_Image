import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from get_loader import imageCaptionsDataset
from models import EncoderDecoder
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms



resize_transform = transforms.Resize((299, 299))    # for the pretrain inception module

def collate_fn(data, tokenizer):
    image, caption = zip(*data)  # need to return too the attentions mask
    caption_lengths = [len(i) for i in caption]
    pad_token_id = tokenizer.pad_token_id
    caption = pad_sequence(caption, padding_value=pad_token_id)
    return torch.stack([resize_transform(img) for img in image]), caption, caption_lengths


def train(batch_size=32, training_prediction="recursive"):
    """
    :param batch_size:
    :param training_prediction: implementation of two methods: recursive and teacher forcing
    :return:
    """
    # parameters
    learning_rate = 0.001
    epochs = 2
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    print(len(tokenizer))
    dataset = imageCaptionsDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))
    enc_dec = EncoderDecoder(tokenizer)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(enc_dec.parameters(), lr=learning_rate)
    enc_dec.train()
    for epoch in range(epochs):
        for batch in dataloader:
            img, cap, len_cap = batch
            if training_prediction == "recursive":
                enc_dec.recursive_method(img, tokenizer.pad_token_id, max_len=30)
            elif training_prediction == "teacher_forcing":
                pass






if __name__ == '__main__':
    train()