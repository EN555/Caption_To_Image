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
    def __init__(self, embedding_size=100):
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
        self.linear = nn.Linear(50, vocab_len)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input, hidden, caption=None, img=0):
        """
        :param input:
        :param hidden:
        :param caption:
        :param img: bool parameter, check if to use nn.embedding
        :return:
        """
        # teacher forcing
        if not img:
            input = self.embedding(input)
        if caption:
            caption = self.embedding(caption)
            caption = torch.transpose(caption, 0, 1)
            input = torch.cat((input.unsqueeze(1), caption), dim=1)
        output, hidden = self.lstm(input, hidden)
        output = self.dropout(self.linear(output))
        return output, hidden


class EncoderDecoder(nn.Module):
    def __init__(self, tokenizer):
        super(EncoderDecoder, self).__init__()
        self.encdoer = encodeImage()
        self.decoder = decodeImage(len(tokenizer))
        self.tokenizer = tokenizer

    def forward(self, x, caption, len_cap):
        encoding = self.encdoer(x)
        decoding = self.decoder(encoding, caption)
        return decoding


    def recursive_method(self, input, padding_idx, max_len=30):
        hidden = None
        input = self.encdoer(input)
        img = 1
        for idx, _ in enumerate(range(max_len)):
            output, hidden = self.decoder(input=input, hidden=hidden, img=0 if idx == 0 else 1)
            img = 0
            softmax = torch.softmax(output, dim=1)
            output = input



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    enc_dec = EncoderDecoder(len(tokenizer))