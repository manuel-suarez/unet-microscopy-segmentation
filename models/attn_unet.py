import unittest
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from unet import Encoder, DecoderUpBlock, SegmentationLayer


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, name, filters, size, **kwargs):
        super(AttentionBlock, self).__init__(name=name, **kwargs)
        self.conv_x = layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1))
        self.conv_g = layers.Conv2D(filters, kernel_size=(1, 1))
        self.relu = layers.Activation('relu')
        self.sigmoid = layers.Activation('sigmoid')
        self.conv_ag = layers.Conv2D(filters, kernel_size=(1, 1))
        self.upsample = layers.UpSampling2D(size=(size, size), interpolation='trilinear')

    def call(self, inputs):
        x, g = inputs
        x = self.conv_x(x)
        g = self.conv_g(g)
        ag = layers.add([x, g])
        ag = self.relu(ag)
        ag = self.conv_ag(ag)
        attention_coefficients = self.sigmoid(ag)
        scaled_coefficients = self.upsample(attention_coefficients)

        return scaled_coefficients

class DecoderAttention(tf.keras.layers.Layer):
    def __init__(self, name="decoder", batch_norm=True, dropout_rate=0.0, **kwargs):
        super(DecoderAttention, self).__init__(name=name, **kwargs)
        # Attention blocks
        self.attblock4 = AttentionBlock("decoder_attnblock4", filters=512, size= 32)
        self.attblock3 = AttentionBlock("decoder_attnblock3", filters=256, size= 64)
        self.attblock2 = AttentionBlock("decoder_attnblock2", filters=128, size=128)
        self.attblock1 = AttentionBlock("decoder_attnblock1", filters= 64, size=256)
        # Upsampling blocks
        self.upblock4 = DecoderUpBlock("decoder_upblock4", filters=512, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.upblock3 = DecoderUpBlock("decoder_upblock3", filters=256, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.upblock2 = DecoderUpBlock("decoder_upblock2", filters=128, batch_norm=batch_norm, dropout_rate=dropout_rate)
        self.upblock1 = DecoderUpBlock("decoder_upblock1", filters= 64, batch_norm=batch_norm, dropout_rate=dropout_rate)

    def call(self, inputs):
        pass

class UNetAttention(tf.keras.Model):
    def __init__(self, input_shape, num_classes=1, dropout_rate=0.0, batch_norm=True):
        super(UNetAttention, self).__init__()
        # Input layer
        self.inputs = layers.Input(input_shape, dtype=tf.float32)
        # Encoder layer
        self.encoder = Encoder(batch_norm=batch_norm, dropout_rate=dropout_rate)
        # Decoder with attention layer
        self.decoder = DecoderAttention(batch_norm=batch_norm, dropout_rate=dropout_rate)
        # Segmentation Block (1*1 convolutional layer)
        self.segmentation = SegmentationLayer()

    def call(self, inputs, **kwargs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return self.segmentation(x)

class TestDecoderAttention(unittest.TestCase):
    def test_upblock4(self):
        input = tf.random.uniform((1, 8, 8, 1024))
        skip = tf.random.uniform((1, 16, 16, 512))
        model = DecoderAttention().upblock4
        self.assertEqual((1, 16, 16, 512), model([input, skip]).shape)