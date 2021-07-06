#!/bin/bash

set -e

## No Pretrain
#python eval.py --data_dir=./data/NEU-CLS-200/ --model_path=./models/best_resnet18_NEU-200.pth --num_classes=6

## Pretrain with NEU-64
#python eval.py --data_dir=./data/NEU-CLS-200/ --model_path=./models/best_resnet18_NEU-64_NEU-200.pth --num_classes=6
#python eval.py --data_dir=./data/NEU-CLS-200/ --model_path=./models/best_resnet18_NEU-64_NEU-200_fixed.pth --num_classes=6

## Pretrain with ImageNet
#python eval.py --data_dir=./data/NEU-CLS-200/ --model_path=./models/best_resnet18_ImageNet_NEU-200.pth --num_classes=6
#python eval.py --data_dir=./data/NEU-CLS-200/ --model_path=./models/best_resnet18_ImageNet_NEU-200_fixed.pth --num_classes=6

## Pretrain with ImageNet and NEU-64
python eval.py --data_dir=./data/NEU-CLS-200/ --model_path=./models/best_resnet18_ImageNet_NEU-64_NEU-200.pth --num_classes=6
python eval.py --data_dir=./data/NEU-CLS-200/ --model_path=./models/best_resnet18_ImageNet_NEU-64_NEU-200_fixed.pth --num_classes=6