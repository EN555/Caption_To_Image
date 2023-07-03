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
        self.linear = nn.Linear(self.inception.fc.out_features, embedding_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.stack([resize_transform(image) for image in x])
        x, _ = self.inception(x)
        x = self.linear(x)
        return x


class decodeImage(nn.Module):
    def __init__(self, vocab_len):
        super(decodeImage, self).__init__()
        self.embedding = nn.Embedding(vocab_len, 100)
        self.lstm = nn.LSTM(100, 50, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoding, caption):
        caption = self.embedding(caption)
        encoding = torch.transpose(encoding, 0, 1)
        x = caption.unsqueeze(1).expand(caption.size(0), encoding.size(1), caption.size(1))
        x = torch.cat(encoding, x)
        x = x.reshape(caption.size(0), caption.size(1),-1)
        lstm = self.dropout(self.lstm(x))
        return lstm


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_len):
        super(EncoderDecoder, self).__init__()
        self.encdoer = encodeImage()
        self.decoder = decodeImage(vocab_len)

    def forward(self, x, caption, len_cap):
        encoding = self.encdoer(x)
        decoding = self.decoder(encoding, caption)
        return decoding

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    enc_dec = EncoderDecoder(len(tokenizer))