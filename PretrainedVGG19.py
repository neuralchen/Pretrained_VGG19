#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: PretrainedVGG.py
# Created Date: Friday October 30th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 21st December 2020 10:45:25 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import torch
import torch.nn as nn

def GetVGGModel(pretrained_ckpt, output_layer=31):
    """
        pretrained_ckpt:    the file path to pretrained vgg model
        outputlayer:        selected out put layer number
    """
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),            # layer 1
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 2
        nn.Conv2d(3, 64, (3, 3)),           # layer 3
        nn.ReLU(),                          # layer 4 relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 5
        nn.Conv2d(64, 64, (3, 3)),          # layer 6
        nn.ReLU(),                          # layer 7 relu1-2
        nn.MaxPool2d((2, 2), (2, 2), 
                (0, 0), ceil_mode=True),    # layer 8
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 9
        nn.Conv2d(64, 128, (3, 3)),         # layer 10
        nn.ReLU(),                          # layer 11 relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 12
        nn.Conv2d(128, 128, (3, 3)),        # layer 13
        nn.ReLU(),                          # layer 14 relu2-2
        nn.MaxPool2d((2, 2), (2, 2),
                (0, 0), ceil_mode=True),    # layer 15
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 16
        nn.Conv2d(128, 256, (3, 3)),        # layer 17
        nn.ReLU(),                          # layer 18 relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 19
        nn.Conv2d(256, 256, (3, 3)),        # layer 20
        nn.ReLU(),                          # layer 21 relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 22
        nn.Conv2d(256, 256, (3, 3)),        # layer 23
        nn.ReLU(),                          # layer 24 relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 25
        nn.Conv2d(256, 256, (3, 3)),        # layer 26
        nn.ReLU(),                          # layer 27 relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0),# layer 28
                ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 29
        nn.Conv2d(256, 512, (3, 3)),        # layer 30
        nn.ReLU(),                          # layer 31 relu4-1
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 32
        nn.Conv2d(512, 512, (3, 3)),        # layer 33
        nn.ReLU(),                          # layer 34 relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 35
        nn.Conv2d(512, 512, (3, 3)),        # layer 36
        nn.ReLU(),                          # layer 37 relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 38
        nn.Conv2d(512, 512, (3, 3)),        # layer 39
        nn.ReLU(),                          # layer 40 relu4-4
        nn.MaxPool2d((2, 2), (2, 2), 
                (0, 0), ceil_mode=True),    # layer 41
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 42
        nn.Conv2d(512, 512, (3, 3)),        # layer 43
        nn.ReLU(),                          # layer 44 relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 45
        nn.Conv2d(512, 512, (3, 3)),        # layer 46
        nn.ReLU(),                          # layer 47 relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 48
        nn.Conv2d(512, 512, (3, 3)),        # layer 49
        nn.ReLU(),                          # layer 50 relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 51
        nn.Conv2d(512, 512, (3, 3)),        # layer 52
        nn.ReLU()                           # layer 53 relu5-4
    )
    
    if output_layer<=2:
        out_channle = 3
    elif output_layer<=9:
        out_channle = 64
    elif output_layer<=16:
        out_channle = 128
    elif output_layer<=29:
        out_channle = 256
    else:
        out_channle = 512
    
    vgg.load_state_dict(torch.load(pretrained_ckpt))
    vgg = nn.Sequential(*list(vgg.children())[:output_layer])
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg, out_channle