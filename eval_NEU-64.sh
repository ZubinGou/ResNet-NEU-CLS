#!/bin/bash

set -e
## No Pretrain
#python eval.py --data_dir=./data/NEU-CLS-64/ --model_path=./models/best_resnet18_NEU-64.pth --num_classes=9

## Pretrain with ImageNet
python eval.py --data_dir=./data/NEU-CLS-64/ --model_path=./models/best_resnet18_ImageNet_NEU-64.pth --num_classes=9

[[ 82   0   0   0   0   0]
 [  0  91   0   0   0   0]
 [  0   0 104   0   0   0]
 [  0   0   0  81   0   0]
 [  0   0   0   0  84   0]
 [  0   0   0   0   0  78]]
              precision    recall  f1-score   support

          cr       1.00      1.00      1.00        82
          in       1.00      1.00      1.00        91
          pa       1.00      1.00      1.00       104
          ps       1.00      1.00      1.00        81
          rs       1.00      1.00      1.00        84
          sc       1.00      1.00      1.00        78

    accuracy                           1.00       520
   macro avg       1.00      1.00      1.00       520
weighted avg       1.00      1.00      1.00       520

Acc on test set: 1.000000