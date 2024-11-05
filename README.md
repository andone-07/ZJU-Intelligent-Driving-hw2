# HW2: Fine-tune the object detection model on the autopilot dataset

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

[![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github)]()

This repository stores the assignment 2 code of the Intelligent driving course of Autumn Semester 2024 of the Software School of Zhejiang University.

This project is based on [YOLOv8](https://github.com/ultralytics/ultralytics), adding [EMA](https://arxiv.org/abs/2305.13563v2) attention mechanism and training on [SODA10M](https://soda-2d.github.io/download.html) dataset.


## Table of Contents

- [Project Structure](#project-structure)
- [Install](#install)
- [Data Preprocess](#data-preprocess)
- [Train](#train)
- [Adding EMA](#adding-ema)
- [Result](#result)
- [Download](#download)


## Project Structure

```md
.
├── dataset/
├── images/
├── ultralytics/
├── requirements.txt
├── transform.py
├── test_model.py
└── train.py
```

- `dataset/`: Directory for placing training data SODA10M.
- `ultralytics/`: Directory where the yolov8 model code and related scripts are stored.
- `transform.py`: Convert the data set format.
- `test_model.py`: Script for testing the modified model.
- `train.py`: Script for training the model.

## Install

This code runs in the conda virtual environment.

You can see the libraries you need in the requirements.txt.

```
# First, you need to install Python and Anaconda3

# Create a virtual environment and activate it
conda create -n your_env_name python=3.10
conda activate your_env_name

# install
pip install -r requirements.txt
```

## Data Preprocess

Before training, you need to convert the COCO data set to YOLO using transform.py.
```
python transform.py
```
## Train

You can train the model with the following commands.

```
# before the adjustment
python train.py --data soda.yaml --cfg yolov8.yaml --weights pretrained/yolov8s.pt

# after the adjustment
python train.py --data soda.yaml --cfg yolov8_EMA.yaml --weights pretrained/yolov8s.pt
```

Note: The `license` badge image link at the top of this file should be updated with the correct `:user` and `:repo`.


## Adding EMA

We added the EMA attention mechanism to the YOLOv8 model as follows.

1、Create the EMA.py file in the path .\ultralytics-main\ultralytics\nn.

```
# EMA.py

import torch
from torch import nn

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
```
2、Import the EMA module in the tasks.py file in the same path and add some code to it.
```
# tasks.py

from ultralytics.nn.EMA import EMA
# line 1045
elif m in {EMA}:
            args = [ch[f]]
```
3、Create the yolov8_EMA.yaml file and add the attention mechanism module code to it.
```
# yolov8_EMA.yaml

nc: 6 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 1, EMA, [1024, 8]] # 10 # EMA

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

## Result

#### Before adding EMA attention mechanism, the results of model training and verification are as follows.
- Precision: 0.691
- mAP50: 0.589

![result1](images\result1.png)

#### After adding EMA attention mechanism, the results of model training and verification are as follows.
- Precision: 0.696
- mAP50: 0.59

![result2](images\result2.png)

## Download

The model file download link is below:
[model file: YOLOv8_EMA.pt](#)