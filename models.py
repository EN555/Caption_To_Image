import torchvision.models as models
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer



class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features



class DecoderT5(nn.Module):
    def __init__(self, t5_model_name='t5-small'):
        super(DecoderT5, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    def forward(self, input_features, captions):
        inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_features)
        return outputs.logits
