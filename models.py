import torchvision.models as models
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from transformers import ViTModel


class ViTEncoder(nn.Module):
    def __init__(self, embed_size, pretrained_model="google/vit-base-patch16-224"):
        super(ViTEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model)
        self.linear = nn.Linear(self.vit.config.hidden_size, embed_size)


    def forward(self, images):
        outputs = self.vit(pixel_values=images)
        vit_output = outputs.last_hidden_state
        avg_output = torch.mean(vit_output, dim=1)
        features = self.linear(avg_output)
        return features



class ImageFeatureToToken(nn.Module):
    def __init__(self, feature_dim, vocab_size, sequence_length):
        super(ImageFeatureToToken, self).__init__()
        self.linear = nn.Linear(feature_dim, vocab_size * sequence_length)
        self.vocab_size - vocab_size
        self.sequence_length = sequence_length

    def forward(self, image_features):
        # image_features shape: (batch_size, feature_dim)
        token_logits = self.linear(image_features)
        # Reshape to (batch_size, sequence_length, vocab_size)
        token_logits = token_logits.view(-1, self.sequence_length, self.vocab_size)
        token_ids = torch.argmax(token_logits, dim=-1)
        return token_ids




class DecoderT5(nn.Module):
    def __init__(self, t5_model_name='t5-small'):
        super(DecoderT5, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.image_to_tokens = ImageFeatureToToken()

    def forward(self, input_features, captions):
        inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids, attention_mask = inputs.input_ids.long(), inputs.attention_mask
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=self.image_to_tokens(input_features))
        return outputs.logits
