import keras
from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import Sequence
from qkeras import *
# from dataset_utils import quantize_manual

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def hard_quantize(x, levels, thresholds):
        x_reshaped = tf.expand_dims(x, axis=-1) 
        is_grt_th = x_reshaped > thresholds 
        indices = tf.reduce_sum(tf.cast(is_grt_th, dtype=tf.int32), axis=-1) 
        return tf.gather(levels, indices)

def var_network(var, hidden=10, output=2):
    var = Flatten()(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def conv_network(var, kernel_size=3, n_filters=5):
    var = QSeparableConv2D(
        n_filters, kernel_size,
        depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = QConv2D(
        n_filters,1,
        kernel_quantizer=quantized_bits(4, 0, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)    
    return var

def CreateModel(shape, n_filters, pool_size, conv_kernel_size, mean_filter=False, thresh = False):
    x_base = x_in = Input(shape)
    
    if mean_filter:
        stack = AveragePooling2D(
            pool_size=(mean_filter,mean_filter), 
            strides = (1, 1), 
            padding = "same",
             #data_format = "channels_first"
        )(x_base)
        if thresh:
            apply_thresh = lambda x: tf.where(x < thresh, tf.zeros_like(x), x)
            stack = Lambda(function=apply_thresh)(stack)
        stack = conv_network(stack, kernel_size=conv_kernel_size)
    else:
        stack = conv_network(x_base, kernel_size=conv_kernel_size)
    
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model

def encoder_network(shape, var, kernel_size=3, n_filters=5, pool_size=2):
    var = QSeparableConv2D(
        n_filters, kernel_size, padding = 'same',
        # depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        # pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        # bias_quantizer=quantized_bits(4, 0, alpha=1),
        # depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        # pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        # activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=pool_size, 
        padding="same", 
        data_format='channels_last',
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = QConv2D(
        n_filters,kernel_size, padding = 'same',
        # kernel_quantizer=quantized_bits(4, 0, alpha=1),
        # bias_quantizer=quantized_bits(4, 0, alpha=1),
        # kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        # activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)  
    var = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=pool_size, 
        padding="same", 
        data_format='channels_last',
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)  
    return var

def decoder_network(shape, var, kernel_size=3, n_filters=5, pool_size=2, input_quant=[247.80, 668.41, 1662.85]):
    var = Conv2DTranspose(
        n_filters, 1, padding = 'same', strides = pool_size,
    )(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)
    var = Conv2DTranspose(
        n_filters, kernel_size,
        strides = pool_size,
        padding="same",
    )(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)    
    var = Conv2D(shape[-1], (pool_size, pool_size), padding="same")(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)
    if input_quant is not None:
        # quantize = lambda x: quantize_manual(x, charge_levels = input_quant, quant_values =[0,1,2,3])
        quantize = lambda x: hard_quantize(x, levels = [0.0,1.0,2.0,3.0], thresholds = input_quant)
        var = Lambda(function=quantize, dtype=tf.float32)(var)
    return var

def decoder_network_v2(shape, var, kernel_size=3, n_filters=5, pool_size=2):
    var = Conv2D(
        n_filters, kernel_size, padding = 'same', #strides = None,
    )(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)
    var = UpSampling2D(pool_size)(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)
    var = Conv2D(
        n_filters, kernel_size,
        # strides = None,
        padding="same",
    )(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)
    var = UpSampling2D(pool_size)(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)    
    var = Conv2D(shape[-1], (pool_size, pool_size), padding="same")(var)
    var = QActivation("quantized_tanh(16, 0, 1)")(var)
    return var

def CreateAutoEncoder(shape, n_filters=5, pool_size=2, conv_kernel_size=3, decoder_filters = 5, mean_filter=False, thresh = False, input_quant = None):
    x_base = x_in = Input(shape)
    stack = encoder_network(shape, x_base, n_filters=n_filters, pool_size=pool_size, kernel_size = conv_kernel_size)
    stack = decoder_network(shape, stack, n_filters=decoder_filters, pool_size=pool_size, kernel_size = conv_kernel_size, input_quant=input_quant)
    # print(type(stack))
    model = Model(inputs=x_in, outputs=stack)
    return model