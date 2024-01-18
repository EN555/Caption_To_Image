import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_data
from transformers import AutoTokenizer
import csv
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os


def parse_captions():
    captions_file = "data/captions.txt"
    image_captions = {}
    with open(captions_file, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            image_file, caption = row
            if image_file in image_captions:
                image_captions[image_file].append(caption)
            else:
                image_captions[image_file] = [caption]
    return image_captions


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, image_captions):
        self.image_dir = image_dir
        self.image_captions = image_captions
        self.transform = transform
        self.image_files = list(image_captions.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        captions = self.image_captions[image_file]
        image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
        image = self.transform(image)
        return image, captions[0]  # it's just for simplicity


if __name__ == '__main__':
    image_captions = parse_captions()
    dataset = ImageCaptionDataset("data/images", image_captions)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("work!")