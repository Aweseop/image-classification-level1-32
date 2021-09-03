import os
import multiprocessing
from typing import Any
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *
from PIL import Image

from base_dataset import MaskSKFSplitByProfileDataset, GenderLabels, AgeLabels
from label_changer import labelChanger


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


class KSAgeAdjDataset(MaskSKFSplitByProfileDataset):
    num_classes = 3

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        image_transform = self.transform(image)

        age_label = self.get_age_label(index)

        return image_transform, age_label
    
    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)

                    id, gender, race, age = profile.split("_")

                    mask_label, gender, age = labelChanger(profile,_file_name, gender, age, self._file_names)

                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(self._adjust_age(int(age)) if phase == 'train' else age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def _adjust_age(self, age):
        if age >= 59:
            return 60
        else:
            return age

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



class KSTestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.mean = mean
        self.std = std

        # self.transform = transforms.Compose([
        #     Resize(resize, Image.BILINEAR),
        #     ToTensor(),
        #     Normalize(mean=mean, std=std),
        # ])
        

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

    def set_transform(self, transform):
        self.transform = transform