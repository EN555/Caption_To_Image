import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from get_loader import *
from models import *

# def prediction(image, )

def train(batch_size=32, training_prediction="recursive"):
    """
    :param batch_size:
    :param training_prediction: implementation of two methods: recursive and teacher forcing
    :return:
    """
    batch_size = 32
    image_captions = parse_captions()
    dataset = ImageCaptionDataset("data/images", image_captions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    encoder = ViTEncoder(100)
    decoder = DecoderT5()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=decoder.tokenizer.pad_token_id)

    for images, captions in dataloader:
        optimizer.zero_grad()
        image_features = encoder(images)

        captions_input = [" ".join(caption.split(" ")[:-1]) for caption in captions]  # Inputs exclude the last token
        captions_target = [" ".join(caption.split(" ")[1:]) for caption in captions]  # Targets exclude the first token

        logits = decoder(image_features, captions_input)
        loss = criterion(logits.view(-1, logits.size(-1)), captions_target.contiguous().view(-1))
        loss.backward()
        optimizer.step()

        print(f"Training Loss: {loss.item()}")


if __name__ == '__main__':
    train()