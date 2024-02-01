import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from unittest import TestCase

def convolutional_block(input, filters, batch_norm=False):
    conv = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)

    return conv

def encoder_block(input, filters, batch_norm=False, dropout_rate=0.0):
    # First convolutional block
    conv = convolutional_block(input, filters, batch_norm)
    # Second convolutional block
    conv = convolutional_block(conv, filters, batch_norm)
    if dropout_rate > 0:
        conv = layers.Dropout(dropout_rate)(conv)

    return conv

def decoder_block(input, skip, filters, batch_norm=False, dropout_rate=0.0):
    # Upsampling
    upsample = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(input)
    # TODO test concatenation with axis=3
    concat = layers.concatenate([upsample, skip])
    conv = convolutional_block(concat, filters, batch_norm)
    if dropout_rate > 0:
        conv = layers.Dropout(dropout_rate)(conv)

    return conv

def UNet(input_shape=(256, 256, 3), batch_norm=True, dropout_rate=0.0):
    inputs = layers.Input(input_shape)
    # Encoder path
    block1 = encoder_block(inputs, 64, batch_norm, dropout_rate)
    pool1  = layers.MaxPooling2D(pool_size=(2,2))(block1)
    block2 = encoder_block(pool1, 128, batch_norm, dropout_rate)
    pool2  = layers.MaxPooling2D(pool_size=(2,2))(block2)
    block3 = encoder_block(pool2, 256, batch_norm, dropout_rate)
    pool3  = layers.MaxPooling2D(pool_size=(2,2))(block3)
    block4 = encoder_block(pool3, 512, batch_norm, dropout_rate)
    pool4  = layers.MaxPooling2D(pool_size=(2,2))(block4)
    # Bottleneck
    block5 = encoder_block(pool4, 1024, batch_norm, dropout_rate)
    # Decoder path
    upblock4 = decoder_block(block5, block4, 512, batch_norm, dropout_rate)
    upblock3 = decoder_block(upblock4, block3, 236, batch_norm, dropout_rate)
    upblock2 = decoder_block(upblock3, block2, 128, batch_norm, dropout_rate)
    upblock1 = decoder_block(upblock2, block1, 64, batch_norm, dropout_rate)
    # Classification layer
    outputs = layers.Conv2D(2, kernel_size=(1,1))(upblock1)
    outputs = layers.BatchNormalization(axis=3)(outputs)
    outputs = layers.Activation('sigmoid')(outputs)

    model = keras.Model(inputs, outputs)
    return model