#!/bin/bash
TRAINFILE="./train.py" # train.py의 경로를 지정해주세요 

# ----- Custom Arg -----
# 따로 수정사항 없으면 주석처리 ( ctrl+/ )

seed="32" 
epochs="5"
dataset="MaskSKFSplitByProfileDataset"
augmentation="ISO_FLIP_Albumentation"
# resize="260 200" #띄어쓰기 1개로 구분해주세요
batch_size="64" # input batch size for training (default: 64)
valid_batch_size="200" 
model="ResNet50"
optimizer="Adam"
lr="3e-5"

criterion="ff"
# lr_decay_step="5"
# log_interval="20"
name="if" # Input custom name
data_dir="/opt/ml/input/data/train/images"
# model_dir="./model"
# ----------------------

cmd=""

function addCmd() {

	if [ -n "$2" ]; then
  	cmd+="--$1 $2 $3 "
	fi
}

addCmd "seed" ${seed:-""}
addCmd "epochs" ${epochs:-""}
addCmd "dataset" ${dataset:-""}
addCmd "augmentation" ${augmentation:-""}
addCmd "resize" ${resize:-""}
addCmd "batch_size" ${batch_size:-""}
addCmd "valid_batch_size" ${valid_batch_size:-""}
addCmd "model" ${model:-""}
addCmd "optimizer" ${optimizer:-""}
addCmd "lr" ${lr:-""}
addCmd "val_ratio" ${val_ratio:-""}
addCmd "criterion" ${criterion:-""}
addCmd "lr_decay_step" ${lr_decay_step:-""}
addCmd "log_interval" ${log_interval:-""}
addCmd "name" ${name:-""}
addCmd "data_dir" ${data_dir:-""}
addCmd "model_dir" ${model_dir:-""}

python3 $TRAINFILE $cmd