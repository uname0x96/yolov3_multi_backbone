#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-07-11 23:37:51
#   Description :
#
#================================================================

import tensorflow as tf
import core.common as common
# import common
from core.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
# from resnet import ResNet50 

#Resnet architecture
def resnet(input_data, num_layer):
    if num_layer == 18:
        restnet = ResNet18() 
    if num_layer == 34:
        restnet = ResNet34() 
    if num_layer == 50:
        restnet = ResNet50()  
    if num_layer == 101:
        restnet = ResNet101()
    if num_layer == 152:
        restnet = ResNet152()

    output_resnet = restnet.call(input_data, training=True)
    
    route_1 = common.convolutional(output_resnet, (3, 3, 2048, 256), downsample=False)

    for _ in range(8):
        route_1 = common.residual_block(route_1, 256, 128, 256)

    route_2 = common.convolutional(output_resnet, (3, 3, 2048, 512), downsample=True)

    for _ in range(8):
        route_2 = common.residual_block(route_2, 512, 256, 512)

    route_3 = common.convolutional(route_2, (3, 3, 512, 1024), downsample=True)

    for _ in range(4):
        route_3 = common.residual_block(route_3, 1024, 512, 1024)

    return route_1, route_2, route_3

# Darnet53 architecture
def darknet53(input_data):

    #conv2d Filters=32 Size=3x3 Strike=1
    input_data = common.convolutional(input_data, (3, 3,  3,  32)) # 256
    #conv2d Filters=64 Size=3x3 Strike=2
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True) # 128

    # The first block residual
    for _ in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64)

    input_data = common.convolutional(input_data, (3, 3,  64, 128), downsample=True) # 64

    for _ in range(2):
        input_data = common.residual_block(input_data, 128,  64, 128)

    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True) # 32

    for _ in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256)

    route_1 = input_data # 256
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True) # 16

    for _ in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512)

    route_2 = input_data #512
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True) # 8

    for _ in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024)

    print('route_1: ', route_1.shape)
    print('route_2: ', route_2.shape)
    print('input_data: ', input_data.shape)

    return route_1, route_2, input_data # 52 | 26 | 13


if __name__ == "__main__":
    # input_data = tf.random.uniform([1, 416, 416, 3], minval=0, maxval=10, dtype='float32')
    # darknet53(input_data)

    input_data = tf.random.uniform([1, 416, 416, 3], minval=0, maxval=10, dtype='float32')
    resnet(input_data)