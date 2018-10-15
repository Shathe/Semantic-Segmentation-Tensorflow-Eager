import tensorflow as tf
from tensorflow.keras import layers, regularizers
import copy


class Segception(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(Segception, self).__init__(**kwargs)
        base_model = tf.keras.applications.xception.Xception(include_top=False, weights=weights,
                                                             input_shape=input_shape, pooling='avg')
        output_1 = base_model.get_layer('block2_sepconv2_bn').output
        output_2 = base_model.get_layer('block3_sepconv2_bn').output
        output_3 = base_model.get_layer('block4_sepconv2_bn').output
        output_4 = base_model.get_layer('block13_sepconv2_bn').output
        output_5 = base_model.get_layer('block14_sepconv2_bn').output
        outputs = [output_5, output_4, output_3, output_2, output_1]

        self.model_output = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        # Decoder
        self.adap_encoder_1 = EncoderAdaption(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_2 = EncoderAdaption(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_3 = EncoderAdaption(filters=256, kernel_size=3, dilation_rate=1)
        self.adap_encoder_4 = EncoderAdaption(filters=128, kernel_size=3, dilation_rate=1)
        self.adap_encoder_5 = EncoderAdaption(filters=64, kernel_size=3, dilation_rate=1)

        self.decoder_conv_1 = FeatureGeneration(filters=256, kernel_size=3, dilation_rate=2, blocks=5)
        self.decoder_conv_2 = FeatureGeneration(filters=128, kernel_size=3, dilation_rate=2, blocks=5)
        self.decoder_conv_3 = FeatureGeneration(filters=64, kernel_size=5, dilation_rate=1, blocks=4)
        self.decoder_conv_4 = FeatureGeneration(filters=64, kernel_size=3, dilation_rate=1, blocks=2)
        self.aspp = ASPP(filters=64, kernel_size=3)

        self.conv_logits = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)


    def call(self, inputs, training=None, mask=None):

        outputs = self.model_output(inputs, training=training)
        # add activations to the ourputs of the model
        for i in range(len(outputs)):
            outputs[i] = layers.LeakyReLU(alpha=0.3)(outputs[i])

        x = self.adap_encoder_1(outputs[0], training=training)
        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_2(outputs[1], training=training), x) #512
        x = self.decoder_conv_1(x, training=training) #256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_3(outputs[2], training=training), x)#256
        x = self.decoder_conv_2(x, training=training) #256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_4(outputs[3], training=training), x)#128
        x = self.decoder_conv_3(x, training=training) #128
        x_aspp = self.aspp(x, training=training, operation='sum') #128
        x += x_aspp

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_5(outputs[4], training=training), x)  # 64
        x = self.decoder_conv_4(x, training=training)  # 64
        x = upsampling(x, scale=2)

        # inputs last features, output: final output with nor rfienemtn
        x = self.conv_logits(x)

        return x




'''
The input has to be the non refined segmentation concatenated with the RGB
'''
class RefineNet(tf.keras.Model):
    def __init__(self, num_classes,  base_model, **kwargs):
        super(RefineNet, self).__init__(**kwargs)
        # mirar si cabe en memoria RefineNet
        # usar esta pero con restore self.model_output.variables y el call es con rgb y segmentacion
        # poner que justo esas a no optimizar, solo las nuevas
        self.base_model = base_model

        # Decoder
        self.conv = FeatureGeneration(filters=32, kernel_size=3, dilation_rate=1, blocks=1)
        self.conv_logits = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)


    def call(self, inputs, training=None, mask=None, iterations=1):
        # get non refined segmentation
        segmentation = self.base_model(inputs, training=training)

        for i in xrange(iterations):
            x = tf.concat([inputs, segmentation], -1)
            x = self.conv(x, training=training)
            segmentation = self.conv_logits(x)

        return segmentation

class Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1):
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = conv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, training=None, activation=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.LeakyReLU(alpha=0.3)(x)

        return x

class DepthwiseConv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(DepthwiseConv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = separableConv(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = layers.LeakyReLU(alpha=0.3)(x)

        return x

class Transpose_Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1):
        super(Transpose_Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = layers.LeakyReLU(alpha=0.3)(x)

        return x
 

def upsampling(inputs, scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale], align_corners=True)

def reshape_into(inputs, input_to_copy):
    return tf.image.resize_bilinear(inputs, [input_to_copy.get_shape()[1].value,
                                          input_to_copy.get_shape()[2].value], align_corners=True)

# convolution
def conv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Traspose convolution
def transposeConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=False):
    return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0003),
                                  dilation_rate=dilation_rate)

# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0003), pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)



class ShatheBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size,  strides=1, dilation_rate=1, bottleneck=2):
        super(ShatheBlock, self).__init__()

        self.filters = filters*bottleneck
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv1 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv2 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv3 = Conv_BN(filters, kernel_size=1)


    def call(self, inputs, training=None):
        x = self.conv(inputs, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x + inputs


class ASPP(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(ASPP, self).__init__()

        self.conv1 = DepthwiseConv_BN(filters, kernel_size=1, dilation_rate=1)
        self.conv2 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=4)
        self.conv3 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=8)
        self.conv4 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=16)
        self.conv5 = Conv_BN(filters, kernel_size=1)


    def call(self, inputs, training=None, operation='concat'):
        feature_map_size = tf.shape(inputs)
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
        image_features = self.conv1(image_features, training=training)
        image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))
        x1 = self.conv2(inputs, training=training)
        x2 = self.conv3(inputs, training=training)
        x3 = self.conv4(inputs, training=training)
        if 'concat' in operation:
            x = self.conv5(tf.concat((image_features, x1, x2, x3), axis=3), training=training)
        else:
            x = image_features + x1 + x2 + x3

        return x




class EncoderAdaption(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1,  strides=1):
        super(EncoderAdaption, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = Conv_BN(filters, kernel_size=1)
        self.conv2 = ShatheBlock(filters, kernel_size=kernel_size, dilation_rate=dilation_rate)

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        return x


class FeatureGeneration(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, blocks=3):
        super(FeatureGeneration, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides


        self.conv0 = Conv_BN(self.filters, kernel_size=1)
        self.blocks = []
        for n in xrange(blocks):
            self.blocks = self.blocks + [ShatheBlock(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)]


    def call(self, inputs, training=None):

        x = self.conv0(inputs, training=training)
        for block in self.blocks:
            x = block(x, training=training)

        return x
