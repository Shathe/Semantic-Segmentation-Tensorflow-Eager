import tensorflow as tf
from tensorflow.keras import layers, regularizers


class Segception(tf.keras.Model):
    def __init__(self, num_classes, alpha=1, input_shape=(None, None, 3), **kwargs):
        super(Segception, self).__init__(**kwargs)
        base_model = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet',
                                                             input_shape=input_shape, pooling='avg')
        output_1 = base_model.get_layer('block2_sepconv2_bn').output
        output_2 = base_model.get_layer('block3_sepconv2_bn').output
        output_3 = base_model.get_layer('block4_sepconv2_bn').output
        output_4 = base_model.get_layer('block13_sepconv2_bn').output
        output_5 = base_model.get_layer('block14_sepconv2_bn').output
        outputs = [output_5, output_4, output_3, output_2, output_1]

        self.model_output = tf.keras.Model(inputs=base_model.input, outputs=outputs)



        # Decoder
        self.adap_encoder_0 = EncoderAdaption(filters=768, kernel_size=3, cardinality=32, bottleneck_factor=2)
        self.decoder_conv_0 = FeatureGeneration(filters=768, kernel_size=3, cardinality=32, bottleneck_factor=2)
        self.conv0 = Conv_BN(512, kernel_size=1)
        self.conv0_1 = Conv_BN(256, kernel_size=1)
        self.conv0_2 = Conv_BN(128, kernel_size=1)
        self.conv0_3 = Conv_BN(64, kernel_size=1)
        self.adap_encoder_1 = EncoderAdaption(filters=512, kernel_size=3, cardinality=32, bottleneck_factor=2)
        self.decoder_conv_1 = FeatureGeneration(filters=512, kernel_size=3)
        self.conv1 = Conv_BN(256, kernel_size=1)
        self.conv1_1 = Conv_BN(128, kernel_size=1)
        self.conv1_2 = Conv_BN(64, kernel_size=1)
        self.adap_encoder_2 = EncoderAdaption(filters=256, kernel_size=3, cardinality=32, bottleneck_factor=2)
        self.decoder_conv_2 = FeatureGeneration(filters=256, kernel_size=3)
        self.conv2 = Conv_BN(128, kernel_size=1)
        self.conv2_1 = Conv_BN(64, kernel_size=1)
        self.adap_encoder_3 = EncoderAdaption(filters=128, kernel_size=3, cardinality=32, bottleneck_factor=2)
        self.decoder_conv_3 = FeatureGeneration(filters=128, kernel_size=3)
        self.conv3 = Conv_BN(64, kernel_size=1)
        self.adap_encoder_4 = EncoderAdaption(filters=64, kernel_size=3, cardinality=32, bottleneck_factor=2)
        self.decoder_conv_4 = FeatureGeneration(filters=64, kernel_size=3)
        self.conv_logits = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)


    def call(self, inputs, training=None, mask=None):

        outputs = self.model_output(inputs, training=training)
        # add activations to the ourputs of the model
        for i in xrange(len(outputs)):
            outputs[i] = layers.LeakyReLU(alpha=0.3)(outputs[i])



        enc_adap_0 = self.adap_encoder_0(outputs[0], training=training)
        enc_adap_1 = self.adap_encoder_1(outputs[1], training=training)
        enc_adap_2 = self.adap_encoder_2(outputs[2], training=training)
        enc_adap_3 = self.adap_encoder_3(outputs[3], training=training)
        enc_adap_4 = self.adap_encoder_4(outputs[4], training=training)

        features_0 = self.decoder_conv_0(enc_adap_0, training=training)
        x = self.conv0(features_0, training=training)
        x = upsampling(x, scale=2)
        x = x + CopyShape()(enc_adap_1, x)



        features_1 = self.decoder_conv_1(x, training=training)
        x = self.conv1(features_1, training=training)
        x = upsampling(x, scale=2)
        x = x + CopyShape()(enc_adap_2, x) + CopyShape()(self.conv0_1(features_0, training=training), x)



        features_2 = self.decoder_conv_2(x, training=training)
        x = self.conv2(features_2, training=training)
        x = upsampling(x, scale=2)
        x = x + CopyShape()(enc_adap_3, x) + CopyShape()(self.conv0_2(features_0, training=training), x) + CopyShape()(self.conv1_1(features_1, training=training), x)


        features_3 = self.decoder_conv_3(x, training=training)
        x = self.conv3(features_3, training=training)
        x = upsampling(x, scale=2)
        x = x + CopyShape()(enc_adap_4, x) + CopyShape()(self.conv0_3(features_0, training=training), x) + CopyShape()(self.conv1_2(features_1, training=training), x) + CopyShape()(self.conv2_1(features_2, training=training), x)

        features_4 = self.decoder_conv_4(x, training=training)

        x = self.conv_logits(features_4)
        x = upsampling(x, scale=2)

        '''
        You could return several outputs, even intermediate outputs
        '''
        return x




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
    def __init__(self, filters, kernel_size, strides=1):
        super(DepthwiseConv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = separableConv(filters=filters, kernel_size=kernel_size, strides=strides)
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


class CopyShape(tf.keras.Model):
    def __init__(self):
        super(CopyShape, self).__init__()

    def call(self, inputs, input_to_copy, training=None):
        x = tf.image.resize_bilinear(inputs, [input_to_copy.get_shape()[1].value,
                                              input_to_copy.get_shape()[2].value], align_corners=True)
        return x







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


 

class GroupConvoution(tf.keras.Model):
    def __init__(self, filters, kernel_size, cardinality=32, strides=1, dilation_rate=1):
        super(GroupConvoution, self).__init__()
        assert not filters % cardinality

        self.filters = int(filters / cardinality) 
        self.kernel_size = kernel_size
        self.cardinality = cardinality
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.convs = []
        self.bns = []
        for c in range(cardinality):
            self.convs = self.convs + [separableConv(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, dilation_rate=self.dilation_rate)]
            self.bns = self.bns + [layers.BatchNormalization(epsilon=1e-3, momentum=0.993)]


    def call(self, inputs, training=None):

        list_to_concatenate = []

        for conv, bn, c in zip(self.convs, self.bns, xrange(self.cardinality)):
            x = inputs[:, :, :, c * self.filters:(c + 1) * self.filters]
            list_to_concatenate = list_to_concatenate + [layers.LeakyReLU(alpha=0.3)(bn(conv(x), training=training))]

        x = tf.concat(list_to_concatenate, axis=-1)

        return x


class ResNeXtBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, cardinality=32, bottleneck_factor=2, strides=1, dilation_rate=1):
        super(ResNeXtBlock, self).__init__()
        assert not (filters // bottleneck_factor) % cardinality

        self.filters = filters
        self.kernel_size = kernel_size
        self.cardinality = cardinality
        self.strides = strides
        self.reduce_filters = int(filters / bottleneck_factor)

        self.conv1 = Conv_BN(self.reduce_filters, kernel_size=1)
        self.conv2 = GroupConvoution(self.reduce_filters, kernel_size=kernel_size, cardinality=cardinality, dilation_rate=dilation_rate)
        self.conv3 = Conv_BN(self.filters, kernel_size=1)

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x + inputs


class EncoderAdaption(tf.keras.Model):
    def __init__(self, filters, kernel_size, cardinality=32, bottleneck_factor=2, strides=1):
        super(EncoderAdaption, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.cardinality = cardinality
        self.strides = strides

        self.conv1 = DepthwiseConv_BN(filters, kernel_size=3)
        self.conv2 = ResNeXtBlock(filters, kernel_size=3, cardinality=cardinality, bottleneck_factor=bottleneck_factor)


    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x


class FeatureGeneration(tf.keras.Model):
    def __init__(self, filters, kernel_size, cardinality=32, bottleneck_factor=2, strides=1):
        super(FeatureGeneration, self).__init__()
        # hacer cosas en plan dilatadas, poolings y tal conectandose entre ellas


        self.filters = filters
        self.kernel_size = kernel_size
        self.cardinality = cardinality
        self.strides = strides
        self.filter_reduced = int(filters / 4)


        self.conv = Conv_BN(self.filter_reduced, kernel_size=1)

        self.conv1 = ResNeXtBlock(self.filter_reduced, kernel_size=3, cardinality=cardinality, bottleneck_factor=bottleneck_factor, dilation_rate=2)
        self.conv2 = ResNeXtBlock(self.filter_reduced, kernel_size=3, cardinality=cardinality, bottleneck_factor=bottleneck_factor, dilation_rate=4)
        self.conv3 = ResNeXtBlock(self.filter_reduced, kernel_size=3, cardinality=cardinality, bottleneck_factor=bottleneck_factor, dilation_rate=8)
        self.conv4 = ResNeXtBlock(self.filter_reduced, kernel_size=3, cardinality=cardinality, bottleneck_factor=bottleneck_factor, dilation_rate=16)

        self.conv5 = ResNeXtBlock(self.filters, kernel_size=3, cardinality=cardinality, bottleneck_factor=bottleneck_factor, dilation_rate=1)



    def call(self, inputs, training=None):

        x = self.conv(inputs, training=training)
        x1 = self.conv1(x, training=training)
        x2 = self.conv2(x1, training=training)
        x3 = self.conv3(x2, training=training)
        x4 = self.conv4(x3, training=training)
        x = tf.concat([x4, x3, x2, x1], axis=-1) + inputs
        x = self.conv5(x, training=training)


        '''

        x1 =  tf.layers.average_pooling2d(x1, pool_size=5, strides=1, padding='same')
        x1 = self.block_1_conv2(x1, training=training)

        x = tf.concat([x4, x3, x2, x1], axis=-1)

        x = self.conv1(x, training=training)
        '''

        return x

