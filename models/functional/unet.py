import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from unittest import TestCase

def conv_bn_relu(input, filters, batch_norm=False):
    conv = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)

    return conv

def convolutional_block(input, filters, batch_norm=False, dropout_rate=0.0):
    # First convolutional block
    conv = conv_bn_relu(input, filters, batch_norm)
    # Second convolutional block
    conv = conv_bn_relu(conv, filters, batch_norm)
    if dropout_rate > 0:
        conv = layers.Dropout(dropout_rate)(conv)

    return conv

def decoder_block(input, skip, filters, batch_norm=False, dropout_rate=0.0):
    # Upsampling
    upsample = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(input)
    # TODO test concatenation with axis=3
    concat = layers.concatenate([upsample, skip], axis=3)
    conv = convolutional_block(concat, filters, batch_norm)
    conv = convolutional_block(conv, filters, batch_norm)
    if dropout_rate > 0:
        conv = layers.Dropout(dropout_rate)(conv)

    return conv

def UNet(input_shape=(256, 256, 1), batch_norm=True, dropout_rate=0.0):
    inputs = layers.Input(input_shape)
    # Encoder Path: 256 -> 128 -> 64 -> 32 -> 16 -> 8
    # Encoder Block1 : convolutional + max pool : 256 -> 128
    block128 = convolutional_block(inputs, 64, batch_norm, dropout_rate)
    pool64   = layers.MaxPooling2D(pool_size=(2,2))(block128)
    # Encoder Block2 : convolutional + max pool : 128 -> 64
    block64  = convolutional_block(pool64, 128, batch_norm, dropout_rate)
    pool32   = layers.MaxPooling2D(pool_size=(2,2))(block64)
    # Encoder Block3 : convolutional + max pool : 64 -> 32
    block32  = convolutional_block(pool32, 256, batch_norm, dropout_rate)
    pool16   = layers.MaxPooling2D(pool_size=(2,2))(block32)
    # Encoder Block4 : convolutional + max pool : 32 -> 16
    block16  = convolutional_block(pool16, 512, batch_norm, dropout_rate)
    pool8    = layers.MaxPooling2D(pool_size=(2,2))(block16)
    # Bottleneck
    block8   = convolutional_block(pool8, 1024, batch_norm, dropout_rate)
    # Decoder path: 8 -> 16 -> 32 -> 64 -> 128 -> 256
    # Decoder Block4 : upsample + concat + convolutional : 8 -> 16
    up_block16 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(block8)
    concat16   = layers.concatenate([up_block16, block16], axis=3)
    dec_block16 = convolutional_block(concat16, 512, batch_norm, dropout_rate)
    # Decoder Block3 : upsample + concat + convolutional : 16 -> 32
    up_block32 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dec_block16)
    concat32 = layers.concatenate([up_block32, block32], axis=3)
    dec_block32 = convolutional_block(concat32, 256, batch_norm, dropout_rate)
    # Decoder Block2 : upsample + concat + convolutional : 32 -> 64
    up_block64 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dec_block32)
    concat64 = layers.concatenate([up_block64, block64], axis=3)
    dec_block64 = convolutional_block(concat64, 128, batch_norm, dropout_rate)
    # Decoder Block1 : upsample + concat + convolutional : 64 -> 128
    up_block128 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dec_block64)
    concat128 = layers.concatenate([up_block128, block128], axis=3)
    dec_block128 = convolutional_block(concat128, 64, batch_norm, dropout_rate)
    # Classification layer
    outputs = layers.Conv2D(1, kernel_size=(1,1))(dec_block128)
    outputs = layers.BatchNormalization(axis=3)(outputs)
    outputs = layers.Activation('sigmoid')(outputs)

    model = keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    model = UNet((256, 256, 1))
    print(model.summary())