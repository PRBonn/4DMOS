#!/usr/bin/env python3
# @file      customminkunet.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved


from .minkunet import MinkUNet14


class CustomMinkUNet(MinkUNet14):
    PLANES = (8, 16, 32, 64, 64, 32, 16, 8)
    INIT_DIM = 8
