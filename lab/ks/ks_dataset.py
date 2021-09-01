import multiprocessing
from typing import Any
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *
from PIL import Image

from base_dataset import MaskSKFSplitByProfileDataset

class KSSample():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im kwansik')

class KSAgeDataset(MaskSKFSplitByProfileDataset):
    num_classes = 3

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        image_transform = self.transform(image)

        age_label = self.get_age_label(index)

        return image_transform, age_label

class KSGenderDataset(MaskSKFSplitByProfileDataset):
    num_classes = 2

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        image_transform = self.transform(image)

        gender_label = self.get_gender_label(index)

        return image_transform, gender_label

class KSMaskDataset(MaskSKFSplitByProfileDataset):
    num_classes = 3
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        image_transform = self.transform(image)

        mask_label = self.get_mask_label(index)

        return image_transform, mask_label



class KSestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.mean = mean
        self.std = std

        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
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

    def set_transform(self, transform):
        self.transform = transform