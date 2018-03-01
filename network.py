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
from datetime import datetime
from optparse import OptionParser


#------------------------------------------------------------------------------------------------------------
def max_feature_map(tensor, netType, name='activation'):
    # tensor is a 4-dimension array
    
    #print('Performing Max-Feature-Map activation ..')
    if netType == 'fc':
        # In case of FC layer, input tensor would be of two dimension.
        # First dimension specifies the number of data units while second one specifies the output unit
        # or the number of neurons in that FC layer. We split it into two parts along dimension 2 (axis=1)
        
        x0, x1 = tf.split(tensor, num_or_size_splits = 2, axis = 1)
        y = tf.maximum(x0, x1)
        
    elif netType == 'conv':
        x0, x1 = tf.split(tensor, num_or_size_splits = 2, axis = 3) # split along the channel dimension
        y = tf.maximum(x0, x1)
        
    return y

#------------------------------------------------------------------------------------------------------------

def maxPool2x2(x, kernel_size, stride_size, deviceId="/gpu:0"):
    with tf.device(deviceId):
        return tf.nn.max_pool(x, ksize=kernel_size,
                        strides=stride_size, padding='SAME')

#------------------------------------------------------------------------------------------------------------
def init_weights(shape, layer, init_type='xavier', deviceId="/gpu:0"):
        
    if init_type == 'truncated_normal':
        #print('.. Truncated_normal weight initialization.')
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1),name=layer+"_W")                
    
    elif init_type == 'xavier':
        #print('..Xavier init')                
        return tf.get_variable(layer+"_W", shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    
    elif init_type == 'orthogonal':        
        #print('****** Orthogonal weight initialization ******')
        return tf.get_variable(layer+"_W", shape=shape, 
                               initializer=tf.orthogonal_initializer(gain=1.0, dtype=tf.float32, seed=None))
    
#------------------------------------------------------------------------------------------------------------
def bias_variable(shape, layer, init_type='xavier',deviceId="/gpu:0"):    
    #print('***** Bias initialization, set to 0 ***** ')
    return tf.Variable(tf.constant(0.0, shape=shape, name=layer+"_B"))
    
#---------------------------------------------------------------------------------------------------------------------                                                                                       
def variable_summaries(var, deviceId="/gpu:0"):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.device(deviceId):    
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))     
            tf.summary.scalar('stddev', stddev)
            
        #tf.summary.scalar('max', tf.reduce_max(var))
        #tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#---------------------------------------------------------------------------------------------------------------------                
def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.elu, init_type='xavier', deviceId="/gpu:0"):
    
    with tf.device(deviceId):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):            
            with tf.variable_scope('weights', reuse=None):
                weights = init_weights([input_dim, output_dim], layer_name, init_type)                
                variable_summaries(weights)
                
            with tf.variable_scope('biases', reuse=None):    
                biases = bias_variable([output_dim], layer_name, init_type)
                variable_summaries(biases)
                        
            with tf.variable_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights)+biases
                tf.summary.histogram('Logits', preactivate)
        
            if act == 'no-activation':
                activations = preactivate
            elif act == 'mfm':
                netType = 'fc'
                activations = max_feature_map(preactivate, netType, name='activation')                
            else:
                activations = act(preactivate, name='activation')
        
            tf.summary.histogram('activations', activations)
            
            return activations, weights, biases

#---------------------------------------------------------------------------------------------------------------------        
def conv_layer(input_tensor,filter_shape,bias_shape,stride_shape,layer_name,padding,act=tf.nn.elu, init_type='xavier',
               deviceId="/gpu:0"):
    #the input_tensor must be a 4dimensional tensor where first two dimension correspond to
    #filter size, third dimension the input filter size and 4th the output filter size
        
    with tf.device(deviceId):    
        with tf.name_scope(layer_name):            
            with tf.variable_scope('weights', reuse=None):
                weights = init_weights(filter_shape, layer_name, init_type)
                variable_summaries(weights)
                    
            with tf.variable_scope('biases', reuse=None):
                biases = bias_variable(bias_shape, layer_name, init_type)   
                variable_summaries(biases)
                        
            with tf.variable_scope('Wx_plus_b'):
                if padding:
                    #print('Shape of tensor: ', input_tensor)
                    #print('Shape of weights: ', weights)
                    conv = tf.nn.conv2d(input_tensor, weights, strides=stride_shape, padding='SAME') + biases
                else:
                    conv = tf.nn.conv2d(input_tensor, weights, strides=stride_shape, padding='VALID') + biases
                    
                tf.summary.histogram('Logits', conv)
                
            if act == 'no-activation':
                activations = conv
                
            elif act == 'mfm':      #For Max-Feature-Map Activation
                netType = 'conv'
                activations = max_feature_map(conv, netType, name='activation')
                
            else:
                activations = act(conv, name='activation')
                        
            tf.summary.histogram('activations', activations)       
                        
            return activations, weights, biases #made this change so that we can add L2 loss regularization on weights
        
#------------------------------------------------------------------------------------------------------------        
def fc_layer_noAct(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.elu, 
                   init_type='xavier', deviceId="/gpu:0"):
    
    with tf.device(deviceId):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):            
            with tf.variable_scope('weights', reuse=None):
                weights = init_weights([input_dim, output_dim], layer_name, init_type)                
                variable_summaries(weights)
                    
            with tf.variable_scope('biases', reuse=None):    
                biases = bias_variable([output_dim], layer_name, init_type)
                variable_summaries(biases)
                    
            with tf.variable_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights)+biases
                tf.summary.histogram('Logits', preactivate)
                
            return preactivate, weights, biases        
        
#------------------------------------------------------------------------------------------------------------        
def conv_layer_noAct(input_tensor, filter_shape, bias_shape, stride_shape, layer_name, act=tf.nn.elu, 
                     init_type='xavier',deviceId="/gpu:0"):
        
    with tf.device(deviceId):    
        with tf.name_scope(layer_name):        
            with tf.variable_scope('weights', reuse=None):
                weights = init_weights(filter_shape, layer_name, init_type)
                variable_summaries(weights)
                    
            with tf.variable_scope('biases', reuse=None):
                biases = bias_variable(bias_shape, layer_name, init_type)   
                variable_summaries(biases)
                        
            with tf.variable_scope('Wx_plus_b'):                
                conv = tf.nn.conv2d(input_tensor, weights, strides=stride_shape, padding='SAME') + biases
                tf.summary.histogram('Logits', conv)
                
            return conv, weights, biases

#------------------------------------------------------------------------------------------------------------        
def applyActivation(logits, layer_name, act):
    with tf.name_scope(layer_name):
        activation=act(logits, 'activation')
        tf.summary.histogram('activation', activation)
        
#---------------------------------------------------------------------------------------------------------------------         
################# Batch Norm layer taken from tutorial
# Source https://github.com/martin-gorner/tensorflow-mnist-tutorial/edit/master/mnist_4.2_batchnorm_convolutional.py

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) 
    # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()        

#---------------------------------------------------------------------------------------------------------------------                
def drop_layer(input_tensor, keep_prob, layer_name, deviceId="/gpu:0"):
    with tf.device(deviceId):
        with tf.name_scope(layer_name):
            tf.summary.scalar('dropout', keep_prob)
            output = tf.nn.dropout(input_tensor, keep_prob)
            return output         
#------------------------------------------------------------------------------------------------------------        