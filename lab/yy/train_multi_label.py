import sys
sys.path.append('/opt/ml/image-classification-level1-32')

import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score

from loss import create_criterion
from lab.yy.yy_utils import *

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- get train, valid dataset
    train_dir = '/opt/ml/input/data/train'

    # seperated train and valid set considerting propotions and people (val_ratio : 0.2)
    train_data = pd.read_csv(os.path.join(train_dir,'train_data_set.csv'))
    validation_data = pd.read_csv(os.path.join(train_dir,'valid_data_set.csv'))

    train_img_paths, valid_img_paths = train_data['image_path'].tolist(), validation_data['image_path'].tolist()
    train_classes, valid_classes = train_data['class'].tolist(), validation_data['class'].tolist()

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)
    train_set = dataset_module(
        img_paths=train_img_paths,
        classes=train_classes
    )
    val_set = dataset_module(
        img_paths=valid_img_paths,
        classes=valid_classes
    )
    num_classes = train_set.num_classes # 8 classes

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)
    validtransform_module = getattr(import_module("dataset"), 'YYValidAugmentation')
    transform = transform_module(
        resize=args.resize,
        mean=train_set.mean,
        std=train_set.std,
    )
    validtransform_module = validtransform_module(
        resize=args.resize,
        mean=train_set.mean,
        std=train_set.std,
    )
    train_set.set_transform(transform)
    val_set.set_transform(validtransform_module)

    # -- data_loader

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: Resnet50
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: multi_label_soft_margin
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
    
    # using differential learning rate
    optimizer = opt_module(
        [
            {'params': model.module.model.conv1.parameters(), 'lr': 1e-4},
            {'params': model.module.model.bn1.parameters(), 'lr': 1e-4},
            {'params': model.module.model.layer1.parameters(), 'lr': 1e-4},
            {'params': model.module.model.layer2.parameters(), 'lr': 1e-3},
            {'params': model.module.model.layer3.parameters(), 'lr': 1e-3},
            {'params': model.module.model.layer4.parameters(), 'lr': 1e-3},
            {'params': model.module.model.fc.parameters(), 'lr': 1e-2}
        ],
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # label smoothing (defalut : False)
            if args.label_smoothing:
                labels = smooth_target(labels, smoothing=args.smoothing, num_classes=num_classes)

            # cutmix (defalut : False)
            if args.cutmix :
                prob = np.random.rand(1)
                if prob < args.cutmix_prob: # cutmix probability (default : 0.5)
                    inputs, labels = cutmix_batch(inputs, labels)

            optimizer.zero_grad()

            outs = model(inputs)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()

            # change multi label to multi class
            preds, originals = get_class_label(outs), get_class_label(labels)

            matches += (preds == originals).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_f1_items = []
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                loss_item = criterion(outs, labels).item()

                # change multi label to multi class
                preds, originals = get_class_label(outs), get_class_label(labels)

                acc_item = (originals == preds).sum().item()
                f1_item = f1_score(originals.cpu().numpy(), preds.cpu().numpy(), average='macro')
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                val_f1_items.append(f1_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            val_f1 = np.sum(val_f1_items) / len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)
            if val_f1 >= best_val_f1:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
                best_val_f1 = val_f1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1: {val_f1:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, f1: {best_val_f1:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    import os

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=32, help='random seed (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--dataset', type=str, default='MultiLabelMaskDataset', help='dataset augmentation type (default: MultiLabelMaskDataset)')
    parser.add_argument('--augmentation', type=str, default='YYAugmentation', help='data augmentation type (default: YYAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[512, 384], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--model', type=str, default='YYResnet50', help='model type (default: YYResnet50)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate (default: 3e-5)')
    parser.add_argument('--criterion', type=str, default='multi_label_soft_margin', help='criterion type (default: multi_label_soft_margin)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler decay step (default: 10)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--cutmix', default=False, help='using cutmix (default:False)')
    parser.add_argument('--cutmix_prob', type= float, default=0.5, help='cutmix probability (default:0.5)')
    parser.add_argument('--label_smoothing', default=False, help='using label_smoothing (default:False)')
    parser.add_argument('--smoothing', default=0.1, help='smoothing (default:0.1)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(model_dir, args)