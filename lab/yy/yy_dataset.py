import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import *
from base_dataset import TestDataset

# Augmentation using in train multi label.py
class YYAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class YYValidAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

# Custom Dataset
class YYMaskDataset(Dataset):
    def __init__(self, img_paths, classes, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.mean = mean
        self.std = std
        self.classes = classes
        self.transform = None

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"
        image = Image.open(self.img_paths[index])
        class_label = self.classes[index]

        if self.transform:
            image = self.transform(image=image)['image']
        return image, class_label

    def __len__(self):
        return len(self.img_paths)

    def set_transform(self, transform):
        self.transform = transform

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

class YYTestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            CenterCrop((400, 400)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

# Dataset for multi label classification
class MultiLabelMaskDataset(Dataset):
    num_classes = 3 + 2 + 3

    def __init__(self, img_paths, classes, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.image_paths = img_paths
        self.mean = mean
        self.std = std
        self.classes = classes
        self.transform = None


    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        class_label = self.get_multi_label(self.classes[index])
        image_transform = self.transform(image)
        return image_transform, class_label

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    def get_multi_label(self, class_label):
        mask_label = (class_label // 6) % 3
        gender_label = (class_label // 3) % 2
        age_label = class_label % 3
        multi_label = np.zeros(8)
        multi_label[0+mask_label] = 1
        multi_label[3+gender_label] = 1
        multi_label[5+age_label] = 1

        return multi_label