#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:42:23 2024

@author: masuareb
"""

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import unittest

#%% Define metrics and losses
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jaccard_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return - dice_coef(y_true, y_pred)

#%% Define auxiliary blocks
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, name, filters, batch_norm=True, dropout_rate=0, **kwargs):
        super(EncoderBlock, self).__init__(name=name, **kwargs)
        # Parameters
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        # Layers
        self.block1_conv1 = layers.Conv2D(filters, (3, 3), padding="same")
        self.batch_norm1  = layers.BatchNormalization(axis=3)
        self.block1_relu1 = layers.Activation("relu")
        self.block1_conv2 = layers.Conv2D(filters, (3, 3), padding="same")
        self.batch_norm2  = layers.BatchNormalization(axis=3)
        self.block1_relu2 = layers.Activation("relu")
        self.block1_drop  = layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = self.block1_conv1(inputs)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = self.block1_relu1(x)

        x = self.block1_conv2(x)
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = self.block1_relu2(x)

        if self.dropout_rate > 0:
            x = self.block1_drop(x)

        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, name="encoder", batch_norm=True, dropout_rate=0, **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        # Blocks
        self.block1 = EncoderBlock("encoder_block1", filters=64, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.block2 = EncoderBlock("encoder_block2", filters=128, batch_norm=batch_norm, dropout_rate=dropout_rate)

    def call(self, inputs):
        x = self.block1(inputs)

        return x

#%% Define networks architectures
class UNet(tf.keras.Model):
    def __init__(self, input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
        '''


        Parameters
        ----------
        input_shape : TYPE
            DESCRIPTION.
        NUM_CLASSES : TYPE, optional
            DESCRIPTION. The default is 1.
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.0.
        batch_norm : TYPE, optional
            DESCRIPTION. The default is True.

        Returns Keras Model
        -------
        None.

        '''
        super(UNet, self).__init__()
        # Input layer
        inputs = layers.Input(input_shape, dtype=tf.float32)
        # Encoder layers
        # Default unet structure + batch_norm + dropout

        # Encoder Block 1
        # Encoder Block 2
        block2_conv1 = layers.Conv2D(128, (3, 3), padding="same")(block1_pool)
        if batch_norm:
            block2_conv1 = layers.BatchNormalization(axis=3)(block2_conv1)
        block2_relu1 = layers.Activation("relu")(block2_conv1)

        block2_conv2 = layers.Conv2D(128, (3, 3), padding="same")(block2_relu1)
        if batch_norm is True:
            block2_conv2 = layers.BatchNormalization(axis=3)(block2_conv2)
        block2_relu2 = layers.Activation("relu")(block2_conv2)

        if dropout_rate > 0:
            block2_relu2 = layers.Dropout(dropout_rate)(block2_relu2)
        block2_pool = layers.MaxPooling2D(pool_size=(2,2))(block2_relu2)
        # Encoder Block 3
        block3_conv1 = layers.Conv2D(256, (3, 3), padding="same")(block2_pool)
        if batch_norm:
            block3_conv1 = layers.BatchNormalization(axis=3)(block3_conv1)
        block3_relu1 = layers.Activation("relu")(block3_conv1)

        block3_conv2 = layers.Conv2D(128, (3, 3), padding="same")(block3_relu1)
        if batch_norm:
            block3_conv2 = layers.BatchNormalization(axis=3)(block3_conv2)
        block3_relu2 = layers.Activation("relu")(block3_conv2)

        if dropout_rate > 0:
            block3_relu2 = layers.Dropout(dropout_rate)(block3_relu2)
        block3_pool = layers.MaxPooling2D(pool_size=(2,2))(block3_relu2)
        # Encoder Block 4
        block4_conv1 = layers.Conv2D(512, (3, 3), padding="same")(block3_pool)
        if batch_norm:
            block4_conv1 = layers.BatchNormalization(axis=3)(block4_conv1)
        block4_relu1 = layers.Activation("relu")(block4_conv1)

        block4_conv2 = layers.Conv2D(512, (3, 3), padding="same")(block4_relu1)
        if batch_norm:
            block4_conv2 = layers.BatchNormalization(axis=3)(block4_conv2)
        block4_relu2 = layers.Activation("relu")(block4_conv2)

        if dropout_rate > 0:
            block4_relu2 = layers.Dropout(dropout_rate)(block4_relu2)
        block4_pool = layers.MaxPooling2D(pool_size=(2,2))(block4_relu2)
        # Encoder Block 5 (Bottleneck)
        block5_conv1 = layers.Conv2D(1024, (3, 3), padding="same")(block4_pool)
        if batch_norm:
            block5_conv1 = layers.BatchNormalization(axis=3)(block5_conv1)
        block5_relu1 = layers.Activation("relu")(block5_conv1)

        block5_conv2 = layers.Conv2D(1024, (3, 3), padding="same")(block5_relu1)
        if batch_norm:
            block5_conv2 = layers.BatchNormalization(axis=3)(block5_conv2)
        block5_relu2 = layers.Activation("relu")(block5_conv2)

        if dropout_rate > 0:
            block5_relu2 = layers.Dropout(dropout_rate)(block5_relu2)

        # Decoder layers (Upsampling)

        # Decoder Layer 4
        upblock4_upsample = layers.UpSampling2D(size=(2,2), data_format="channels_last")(block5_relu2)
        upblock4_concat = layers.concatenate([upblock4_upsample, block4_relu2])
        upblock4_conv1 = layers.Conv2D(512, (3, 3), padding="same")(upblock4_concat)
        if batch_norm:
            upblock4_conv1 = layers.BatchNormalization(axis=3)(upblock4_conv1)
        upblock4_relu1 = layers.Activation("relu")(upblock4_conv1)

        upblock4_conv2 = layers.Conv2D(512, (3, 3), padding="same")(upblock4_relu1)
        if batch_norm:
            upblock4_conv2 = layers.BatchNormalization(axis=3)(upblock4_conv2)
        upblock4_relu2 = layers.Activation("relu")(upblock4_conv2)

        if dropout_rate > 0:
            upblock4_relu2 = layers.Dropout(dropout_rate)(upblock4_relu2)
        # Decoder Layer 3
        upblock3_upsample = layers.UpSampling2D(size=(2,2), data_format="channels_last")(upblock4_relu2)
        upblock3_concat = layers.concatenate([upblock3_upsample, block3_relu2])
        upblock3_conv1 = layers.Conv2D(256, (3, 3), padding="same")(upblock3_concat)
        if batch_norm:
            upblock3_conv1 = layers.BatchNormalization(axis=3)(upblock3_conv1)
        upblock3_relu1 = layers.Activation("relu")(upblock3_conv1)

        upblock3_conv2 = layers.Conv2D(256, (3, 3), padding="same")(upblock3_relu1)
        if batch_norm:
            upblock3_conv2 = layers.BatchNormalization(axis=3)(upblock3_conv2)
        upblock3_relu2 = layers.Activation("relu")(upblock3_conv2)

        if dropout_rate > 0:
            upblock3_relu2 = layers.Dropout(dropout_rate)(upblock3_relu2)
        # Decoder Layer 2
        upblock2_upsample = layers.UpSampling2D(size=(2,2), data_format="channels_last")(upblock3_relu2)
        upblock2_concat = layers.concatenate([upblock2_upsample, block2_relu2])
        upblock2_conv1 = layers.Conv2D(128, (3, 3), padding="same")(upblock2_concat)
        if batch_norm:
            upblock2_conv1 = layers.BatchNormalization(axis=3)(upblock2_conv1)
        upblock2_relu1 = layers.Activation("relu")(upblock2_conv1)

        upblock2_conv2 = layers.Conv2D(128, (3, 3), padding="same")(upblock2_relu1)
        if batch_norm:
            upblock2_conv2 = layers.BatchNormalization(axis=3)(upblock2_conv2)
        upblock2_relu2 = layers.Activation("relu")(upblock2_conv2)

        if dropout_rate > 0:
            upblock2_relu2 = layers.Dropout(dropout_rate)(upblock2_relu2)
        # Decoder Layer 1
        upblock1_upsample = layers.UpSampling2D(size=(2,2), data_format="channels_last")(upblock2_relu2)
        upblock1_concat = layers.concatenate([upblock1_upsample, block1_relu2])
        upblock1_conv1 = layers.Conv2D(64, (3, 3), padding="same")(upblock1_concat)
        if batch_norm:
            upblock2_conv1 = layers.BatchNormalization(axis=3)(upblock1_conv1)
        upblock1_relu1 = layers.Activation("relu")(upblock1_conv1)

        upblock1_conv2 = layers.Conv2D(64, (3, 3), padding="same")(upblock1_relu1)
        if batch_norm:
            upblock1_conv2 = layers.BatchNormalization(axis=3)(upblock1_conv2)
        upblock1_relu2 = layers.Activation("relu")(upblock1_conv2)

        if dropout_rate > 0:
            upblock1_relu2 = layers.Dropout(dropout_rate)(upblock1_relu2)

        # Segmentation Block (1*1 convolutional layer)
        outputs = layers.Conv2D(2, kernel_size=(1,1))(upblock1_relu2)
        outputs = layers.BatchNormalization(axis=3)(outputs)
        outputs = layers.Activation('sigmoid')(outputs)

        # Model
        model = models.Model(inputs, outputs, name="UNet")
        return model

#%% Define unit tests
class TestEncoder(unittest.TestCase):
    def test_block1(self):
        input = tf.random.uniform((1, 128, 128, 3))
        model = Encoder().block1
        self.assertEqual(model(input).shape, (1, 128, 128, 64))

    def test_block2(self):
        input = tf.random.uniform((1, 64, 64, 64))
        model = Encoder().block2
        self.assertEqual(model(input).shape, (1, 64, 64, 128))

class TestUNet(unittest.TestCase):
    def test_unet(self):
        data = tf.random.uniform((1, 128, 128, 3))
        model = UNet(input_shape=(128, 128, 3))
        self.assertEqual(model(data).shape, (1, 128, 128, 2))

#%% Main code
if __name__ == "__main__":

    #unet = UNet(input_shape=(128, 128, 3))
    #print(unet.summary())
    data = tf.random.uniform(shape=(1, 128, 128, 3))
    block1 = Encoder().block1
    print(block1(data).shape)

    #unittest.main()