import tensorflow as tf
import numpy as np
import random
import math
import os
import argparse
import time
import cv2
import math
import sys
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.metrics import confusion_matrix
import math

# Prints the number of parameters of a model
def get_params(model):
	total_parameters = 0
	for variable in model.variables:
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1

		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
	print("Total parameters of the net: " + str(total_parameters) + " == " + str(total_parameters / 1000000.0) + "M")


def preprocess(x, mode='imagenet'):
	if 'imagenet' in mode:
		return tf.keras.applications.resnet50.preprocess_input(x)
	else:
		return x.astype(float32)/127.5 - 1

def lr_decay(lr, init_learning_rate, end_learning_rate, epoch, total_epochs, power=0.9):
	lr.assign((init_learning_rate - end_learning_rate) * math.pow(1 - epoch / 1. / total_epochs, power) + end_learning_rate)


def convert_to_tensors(list_to_convert):
	if list_to_convert != []:
		return [tf.convert_to_tensor(list_to_convert[0])] + convert_to_tensors(list_to_convert[1:])
	else:
		return []


def restore_state(saver, checkpoint):
	try:
		saver.restore(checkpoint)
		print('Model loaded')
	except Exception:
		print('Model not loaded')


def init_model(model, input_shape):
	model._set_inputs(np.zeros(input_shape))

	# Erase the elements if they are from ignore class
def erase_ignore_pixels(labels, predictions, mask):
	indices = tf.squeeze(tf.where(tf.greater(mask, 0)))  # ignore labels
	labels = tf.cast(tf.gather(labels, indices), tf.int64)
	predictions = tf.gather(predictions, indices)

	return labels, predictions


'''
		for scale in scales:
			numpy.resize(a, new_shape)

'''
# get accuracy from the model
def get_metrics(loader, model, n_classes, train=True, flip_inference=False, scales=[1]):
	accuracy = tfe.metrics.Accuracy()
	conf_matrix = np.zeros ((n_classes, n_classes))
	if train:
		samples = len(loader.image_train_list)
	else:
		samples = len(loader.image_test_list)

	# Get train accuracy
	for step in xrange(samples):  # for every batch
		x, y, mask = loader.get_batch(size=1, train=train, augmenter=False)
		x = preprocess(x, mode='imagenet')
		[x, y] = convert_to_tensors([x, y])

		# creates the variable to store the scores
		y_ = convert_to_tensors([np.zeros((x.shape[0],x.shape[1], x.shape[2], n_classes), dtype=np.float32)])[0]

		for scale in scales:
			# scale the image
			x_scaled = tf.image.resize_images(x, (x.shape[1].value*scale, x.shape[2].value*scale), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
			y_scaled = model(x_scaled, training=False)
			#  rescale the output
			y_scaled = tf.image.resize_images(y_scaled, (x.shape[1].value, x.shape[2]), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
			# get scores
			y_scaled = tf.nn.softmax(y_scaled)

			
			if flip_inference:
				#calculates flipped scores
				y_flipped_ = tf.image.flip_left_right(model(tf.image.flip_left_right(x_scaled), training=False))
				# resize to rela scale
				y_flipped_ = tf.image.resize_images(y_flipped_, (x.shape[1].value, x.shape[2]), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
				# get scores
				y_flipped_score = tf.nn.softmax(y_flipped_)

				y_scaled += y_flipped_score

			y_ += y_scaled


		# Rephape
		y = tf.reshape(y, [y.shape[1] * y.shape[2] * y.shape[0], y.shape[3]])
		y_ = tf.reshape(y_, [y_.shape[1] * y_.shape[2] * y_.shape[0], y_.shape[3]])
		mask = tf.reshape(mask, [mask.shape[1] * mask.shape[2] * mask.shape[0]])

		labels, predictions = erase_ignore_pixels(labels=tf.argmax(y, 1),  predictions=tf.argmax(y_, 1), mask=mask)
		accuracy(labels, predictions)
		conf_matrix += confusion_matrix(labels, predictions, labels=range(0, n_classes))
	# get the train and test accuracy from the model
	return accuracy.result(), compute_iou(conf_matrix)



def compute_iou(conf_matrix):
	intersection = np.diag(conf_matrix)
	ground_truth_set = conf_matrix.sum(axis=1)
	predicted_set = conf_matrix.sum(axis=0)
	union = ground_truth_set + predicted_set - intersection
	IoU = intersection / union.astype(np.float32)
	IoU[np.isnan(IoU)] = 0
	return np.mean(IoU)


