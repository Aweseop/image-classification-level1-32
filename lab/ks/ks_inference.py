import argparse

import os
import sys
sys.path.append('/opt/ml/image-classification-level1-32')
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image

from dataset import TestDataset, MaskBaseDataset, KSTestDataset
from util import TextLogger, create_path

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # num_classes = MaskBaseDataset.num_classes  # 18
    # num_classes = 2
    num_classes = args.num_classes  # 18


    # model = load_model(model_dir, num_classes, device).to(device)
    # model.eval()
    age_adj_59_model = load_model(os.path.join('./lab', args.my_name, args.model_dir, "age_cls_adj_59"), 3, device).to(device)
    age_adj_58_model = load_model(os.path.join('./lab', args.my_name, args.model_dir, "age_cls_adj_58"), 3, device).to(device)
    gen_model = load_model(os.path.join('./lab', args.my_name, args.model_dir, "gen_cls"), 2, device).to(device)
    mask_model = load_model(os.path.join('./lab', args.my_name, args.model_dir, "mask_cls"), 3, device).to(device)
    age_adj_59_model.eval()
    age_adj_58_model.eval()
    gen_model.eval()
    mask_model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = KSTestDataset(img_paths, args.resize)
        # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            my_img = Image.open('./profile.jpg')
            images = transform(my_img)
            images = images.to(device)
            # pred = model(images)
            age_adj_59_outs = age_adj_59_model(images)
            age_adj_58_outs = age_adj_58_model(images)
            age_outs = age_adj_59_outs*0.8 + age_adj_58_outs*0.2

            gen_outs = gen_model(images)
            mask_outs = mask_model(images)
            # pred = pred.argmax(dim=-1)
            age_preds = torch.argmax(age_outs, dim=-1)
            gen_preds = torch.argmax(gen_outs, dim=-1)
            mask_preds = torch.argmax(mask_outs, dim=-1)
            print(age_preds)
            print(gen_preds)
            print(mask_preds)
            pred = age_preds + gen_preds*3 + mask_preds*6
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(512, 384), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    # Custom env
    parser.add_argument('--my_name', type=str, default=os.environ.get('MYNAME', 'anonymous'))

    parser.add_argument('--num_classes', type=int, default=3, help='dataset augmentation type (default: MaskSKFSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = os.path.join('./lab', args.my_name, args.model_dir, args.name)
    output_dir = os.path.join('./lab', args.my_name, args.output_dir, args.name)

    # os.makedirs(output_dir, exist_ok=True)
    create_path(output_dir)

    inference(data_dir, model_dir, output_dir, args)
