#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Network.py

This file holds various functions of a neural network within the tensorflow.
We may change it later on when implementing a class.
'''

# Load standard modules
from __future__ import print_function
import sys
import os
import io
import shutil
import numpy as np
import tensorflow as tf

# Load userdefined modules
from network import maxPool2x2
from network import init_weights
from network import bias_variable
from network import variable_summaries
from network import fc_layer
from network import conv_layer
from network import drop_layer
from network import batchnorm
from network import fc_layer_noAct
from network import conv_layer_noAct

#---------------------------------------------------------------------------------------------------------------

def cnnModel3(input_type,trainSize,input_placeholder,activation,init,targets,fftSize,padding,keep_prob1, keep_prob2, keep_prob3,n_layers=2,fc_neurons=100,fc1_neurons=100):
    # This we call as google_small because it uses 100 neurons

    t=trainSize
    trainSize=str(trainSize)+'sec'
    f = 129

    weight_list = list()
    activation_list = list()
    bias_list = list()

    if activation=='mfm':
        if t == 3:
           fc_input=  15*7*8  #f*8   #6448 #1*257*64 = 16448
        
        elif t == 1:
           fc_input = 5*7*8

        in_conv2 = 8
        in_conv3 = 8
        in_conv4 = 8
        in_fc2 = int(fc_neurons/2) #16  #50        
        in_outputLayer = int(fc_neurons/2) #16 #50

    else:

        print('ACtivation is relu')
        if t==3:
           fc_input= 15*7*16   #f*16  #32896 # 1*257*128
        elif t == 1:
           fc_input = 5*7*16 

        in_conv2 = 16
        in_conv3 = 16
        in_conv4 = 16
        in_fc2 = fc_neurons #32  #100
        in_fc3 = fc_neurons #32  #100
        in_outputLayer = fc_neurons #32 #100

    print('======================== CNN ARCHITECTURE ==============================\n')

    #Convolution layer1,2,3
    conv1,w1,b1 = conv_layer(input_placeholder, [1,10,1,16], [16], [1,1,1,1],'conv1',padding,activation,init)
    weight_list.append(w1)
    bias_list.append(b1)
    #print('Conv1 ', iconv1)
    pool1 = maxPool2x2(conv1, [1,2,2,1], [1,2,2,1])
    print('Pool1: ', pool1)

    conv2,w2,b2 = conv_layer(pool1, [1,10,in_conv2,16], [16], [1,1,1,1],'conv2', padding,activation,init)
    weight_list.append(w2)
    bias_list.append(b2)
    #print('Conv2 ', conv2)
    pool2 = maxPool2x2(conv2, [1,2,2,1], [1,2,2,1])
    print('Pool2: ', pool2)

    conv3,w3,b3 = conv_layer(pool2, [1,10,in_conv3,16], [16], [1,1,1,1],'conv3', padding,activation,init)
    weight_list.append(w3)
    bias_list.append(b3)
    #print('Conv3 ', conv3)
    pool3 = maxPool2x2(conv3, [1,5,5,1], [1,5,5,1])
    print('pool3 shape: ', pool3)

    if input_type == 'cqt_spec':
        time_dim = 32
    else:
        time_dim = t*100

    # Dropout on the huge input from Conv layer
    flattened = tf.reshape(pool3, shape=[-1,fc_input])
    dropped_1 = drop_layer(flattened, keep_prob1, 'dropout1')
    fc1,w4,b4, = fc_layer(dropped_1, fc_input, fc_neurons, 'FC_Layer1', activation)
    weight_list.append(w4)
    bias_list.append(b4)

    print('Shape of FC1 = ', fc1.shape)

    
    #Output layer: 2 neurons. One for genuine and one for spoof. Dropout applied first

    print('input to the output layer: ', in_outputLayer)

    dropped_2 = drop_layer(fc1, keep_prob2, 'dropout2')
    output,w7,b7 = fc_layer(dropped_2, in_outputLayer, targets, 'Output_Layer', 'no-activation')  #get raw logits
    print('output layer: shape = ', output.shape)

    weight_list.append(w7)
    bias_list.append(b7)

    print('Output layer shape = ', output.shape)
    print('======================== CNN ARCHITECTURE ==============================\n')

    return fc1, output, weight_list, activation_list, bias_list

