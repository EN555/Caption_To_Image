import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from transformers import AutoTokenizer
import torch

resize_transform = transforms.Resize((299, 299))    # for the pretrain inception module

class encodeImage(nn.Module):
    """
    transfer learning
    """
    def __init__(self, embedding_size=50):
        super(encodeImage, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.requires_grad_(False)
        self.linear = nn.Linear(self.inception.fc.in_features, embedding_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = resize_transform(x)
        x = self.dropout(self.inception(x))
        x = self.linear(x)
        return x


class decodeImage(nn.Module):
    def __init__(self, vocab_len):
        super(decodeImage, self).__init__()
        self.embedding = nn.Embedding(vocab_len, 100)
        self.lstm = nn.LSTM(100, 50)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, caption):
        x = self.embedding(x)
        x = torch.cat(x, caption)
        lstm = self.dropout(self.lstm(x))
        return lstm


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_len):
        super(EncoderDecoder, self).__init__()
        self.encdoer = encodeImage()
        self.decoder = decodeImage(vocab_len)

    def forward(self, x, caption):
        encoding = self.encdoer(x)
        decoding = self.decoder(encoding, caption)
        return decoding

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    enc_dec = EncoderDecoder(len(tokenizer))