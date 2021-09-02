from typing import Any

from torchvision import transforms
from torchvision.transforms import *
import torch
from PIL import Image

import albumentations
import albumentations.pytorch


class BaseAlbumentation:
    def __init__(self, resize, mean, std, **args):
        self.preTransform = albumentations.Compose([
            albumentations.Resize(resize[0], resize[1]),
        ])

        self.customTransform = albumentations.Compose([])
        self.afterTransform = albumentations.Compose([
            albumentations.Normalize(mean, std),
            albumentations.pytorch.ToTensorV2()
        ])

    def __call__(self, image):
        image = ChangeCV()(image)
        image = self.preTransform(image=image)['image']
        image = self.customTransform(image=image)['image']
        # image = ChangePILcolor()(image)
        image = self.afterTransform(image=image)['image']
        return image

class ISO_FLIP_Albumentation(BaseAlbumentation):
    def __init__(self, resize, mean, std, **args):
        super().__init__(resize, mean, std, **args)
        self.customTransform = albumentations.Compose([
            albumentations.Flip(),
            albumentations.ISONoise()
        ])


# --------- Custom transform -----------


import cv2
import numpy as np
class ChangeCV(object):

    def __call__(self, image):
        numpy_image=np.array(image)
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    def __repr__(self):
        return self.__class__.__name__


class ChangePILcolor(object):

    def __call__(self, image):
        numpy_image=np.array(image)
        return cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

    def __repr__(self):
        return self.__class__.__name__
