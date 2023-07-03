import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from get_loader import imageCaptionsDataset
from models import EncoderDecoder
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim




def collate_fn(data, tokenizer):
    image, caption = zip(*data)  # need to return too the attentions mask
    caption_lengths = [len(i) for i in caption]
    pad_token_id = tokenizer.pad_token_id
    caption = pad_sequence(caption, padding_value=pad_token_id)
    return image, caption, caption_lengths


def train(batch_size=32):
    # parameters
    learning_rate = 0.001
    epochs = 2
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    dataset = imageCaptionsDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))
    enc_img = EncoderDecoder(len(tokenizer))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(enc_img.parameters(), lr=learning_rate)
    enc_img.train()
    for epoch in range(epochs):
        for batch in dataloader:
            img, cap, len_cap = batch
            enc_img(img, cap, len_cap)






if __name__ == '__main__':
    train()