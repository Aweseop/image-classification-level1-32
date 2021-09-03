"""Code is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py
https://github.com/naver-ai/relabel_imagenet/blob/main/utils/data_augment.py
https://github.com/hysts/pytorch_mixup/blob/master/utils.py
"""
import torch
import numpy as np

# change classes into one hot vector
def to_one_hot(x, num_classes=18, on_value=1., off_value=0.):
    if len(x.size()) > 1 and x.size(-1) == num_classes:
        # already one-hot form
        return x
    x = x.long().view(-1, 1)

    return torch.full((x.size()[0], num_classes), off_value, device=x.device
                      ).scatter_(dim=1, index=x, value=on_value)

# smooth labeling
# default smoothing 0.1
def smooth_target(target, smoothing=0.1, num_classes=18):
    target *= (1. - smoothing)
    target += (smoothing / num_classes)

    return target

# get random 
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)  # cut ratio
    cut_h = np.int(H * cut_rat)  

   	# y-coordinates of patches
    cy = np.random.randint(H)
		
    # patch edge value
    bbx1 = 0
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = W
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (size[2] * size[3]))

    return bbx1, bby1, bbx2, bby2, lam

# cutmix
def cutmix_batch(images, targets):
    lam = np.random.beta(5.0, 5.0)
    indices = torch.randperm(images.size()[0], device=images.device) # get random index
    target_a = targets
    target_b = targets[indices]
    bbx1, bby1, bbx2, bby2, lam = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2,
                                            bby1:bby2]
    # mix targets
    target = lam * target_a + (1. - lam) * target_b

    return images, target

# mixup
def mixup(images, targets, alpha=1.0):
    indices = torch.randperm(images.size(0))
    images2 = images[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    
    # mix images and targets
    images = images * lam + images2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return images, targets

# change multi label classification(class_nums = 8) logits
# to multi class classification(class_num = 18)
def get_class_label(logits):
    mask_pred = torch.argmax(logits[:,:3], dim=-1)
    gender_pred = torch.argmax(logits[:, 3:5], dim=-1)
    age_pred = torch.argmax(logits[:, 5:], dim=-1)
    pred = mask_pred * 6 + gender_pred * 3 + age_pred

    return pred