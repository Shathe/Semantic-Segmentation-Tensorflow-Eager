import tensorflow as tf
from tensorflow.keras import layers, regularizers, activations



class ResnetFCN(tf.keras.Model):
	def __init__(self, num_classes,  alpha=1, input_shape=(None, None, 3), **kwargs):
		super(ResnetFCN, self).__init__(**kwargs)
		base_model = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=input_shape,  pooling='avg')
		self.resnet_output = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block14_sepconv2_act').output)


		self.blocks_up = []


		# Decoder
		self.blocks_up.append(MBConv_idskip(input_filters=512, filters=192, kernel_size=3, strides=1, filters_multiplier=3, alpha=alpha))
		self.blocks_up.append(MBConv_idskip(input_filters=256, filters=192, kernel_size=3, strides=1, filters_multiplier=3, alpha=alpha))
		self.blocks_up.append(Upsampling())
		self.blocks_up.append(MBConv_idskip(input_filters=256, filters=192, kernel_size=3, strides=1, filters_multiplier=3, alpha=alpha))

		self.blocks_up.append(Upsampling())
		self.blocks_up.append(MBConv_idskip(input_filters=128, filters=192, kernel_size=5, strides=1, filters_multiplier=3, alpha=alpha))

		self.blocks_up.append(Upsampling())
		self.blocks_up.append(MBConv_idskip(input_filters=64, filters=192, kernel_size=5, strides=1, filters_multiplier=3, alpha=alpha))

		self.blocks_up.append(Upsampling())

		self.blocks_up.append(MBConv_idskip(input_filters=32, filters=192, kernel_size=3, strides=1, filters_multiplier=3, alpha=alpha))

		self.blocks_up.append(Upsampling())

		self.conv_logits = conv(filters=num_classes, kernel_size=1, strides=1, use_bias=True)



	def call(self, inputs, training=None, mask=None):
		out = self.resnet_output(inputs, training=training)

		for block in self.blocks_up:
			out = block(out, training=training)

		out = self.conv_logits(out)

		'''
		You could return several outputs, even intermediate outputs
		'''
		return out



class MBConv_idskip(tf.keras.Model):

	def __init__(self, input_filters, filters, kernel_size, strides=1, filters_multiplier=1, alpha=1):
		super(MBConv_idskip, self).__init__()

		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.filters_multiplier = filters_multiplier
		self.alpha = alpha

		self.depthwise_conv_filters = _make_divisible(input_filters) 
		self.pointwise_conv_filters = _make_divisible(filters * alpha)

		#conv1
		self.conv_bn1 = Conv_BN(filters=self.depthwise_conv_filters*filters_multiplier, kernel_size=1, strides=1)

		#depthwiseconv2
		self.depthwise_conv = depthwiseConv(depth_multiplier=1, kernel_size=kernel_size, strides=strides)
		self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)
		self.relu = layers.ReLU(max_value=6)

		#conv3
		self.conv_bn2 = Conv_BN(filters=self.pointwise_conv_filters, kernel_size=1, strides=1)



	def call(self, inputs, training=None):

		x = self.conv_bn1(inputs, training=training)

		x = self.depthwise_conv(x)
		x = self.bn(x, training=training)
		x = self.relu(x)

		x = self.conv_bn2(x, training=training, activation=False)

		
		# Residual/Identity connection if possible
		if self.strides==1 and x.shape[3] == inputs.shape[3]:
			return  layers.add([inputs, x])
		else: 
			return x



class Conv_BN(tf.keras.Model):

	def __init__(self, filters, kernel_size, strides=1):
		super(Conv_BN, self).__init__()

		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides

		self.conv = conv(filters=filters, kernel_size=kernel_size, strides=strides)
		self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)
		self.relu = layers.ReLU(max_value=6)


	def call(self, inputs, training=None, activation=True):

		x = self.conv(inputs)
		x = self.bn(x, training=training)
		if activation:
			x = self.relu(x)

		return x

class Transpose_Conv_BN(tf.keras.Model):



	def __init__(self, filters, kernel_size, strides=1):
		super(Transpose_Conv_BN, self).__init__()

		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides

		self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
		self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)
		self.relu = layers.ReLU(max_value=6)


	def call(self, inputs, training=None, activation=True):
		x = self.conv(inputs)
		x = self.bn(x, training=training)
		if activation:
			x = self.relu(x)

		return x

class Upsampling(tf.keras.Model):



	def __init__(self):
		super(Upsampling, self).__init__()



	def call(self, inputs,  training=None,  size_multiplier=2):
        	x = tf.image.resize_bilinear(inputs, [inputs.get_shape()[1].value*size_multiplier,
					inputs.get_shape()[2].value*size_multiplier], align_corners=True)

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



# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def _make_divisible(v, divisor=8, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


