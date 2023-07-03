import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

resize_transform = transforms.Resize((299, 299))    # for the pretrain inception module

class encodeImage(nn.Module):
    """
    transfer learning
    """
    def __init__(self, embedding_size):
        super(encodeImage, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.requires_grad_(False)
        self.linear = nn.Linear(self.inception.fc.in_features, embedding_size)
        self.relu = F.relu()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = resize_transform(x)
        x = self.dropout(self.relu(self.inception(x)))
        x = self.linear(x)
        return x


class decodeImage(nn.Module):
    def __init__(self, tokenizer):
        super(decodeImage, self).__init__()
        self.embedding = nn.Embedding(len(tokenizer), 100)
        self.lstm = nn.LSTM(100, 50)
        self.relu = F.relu()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        lstm = self.dropout(self.relu(self.lstm(x)))


if __name__ == '__main__':
    encode_img = encodeImage(50)