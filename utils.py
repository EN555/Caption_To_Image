import os
from typing import List, Tuple
from collections import defaultdict
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torch


def get_data(path: str="data")->List[Tuple[torch.tensor, str]]:
    # extract image id and the captions
    img_id_captions = []
    convert_tensor = transforms.ToTensor()
    for caption in open(f"{path}/captions.txt", 'r').readlines()[1:200]:
        img = Image.open(os.path.join(path, "images", caption.split(",")[0]))
        img = convert_tensor(img)
        cap = caption.split(",")[1]
        img_id_captions.append((img, cap))
    return img_id_captions



if __name__ == '__main__':
    all_data = get_data()
    print(all_data[0])
