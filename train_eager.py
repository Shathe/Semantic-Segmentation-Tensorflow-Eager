import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import MnasnetEager
import Loader

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)


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
	print("Total parameters of the net: " + str(total_parameters)+ " == " + str(total_parameters/1000000.0) + "M")

		
def convert_to_tensors(list_to_convert):
	if list_to_convert != []:
		return [tf.convert_to_tensor(list_to_convert[0])] + convert_to_tensors(list_to_convert[1:])
	else:
		return []

# Trains the model for certains epochs on a dataset
def train(loader, model, epochs=5, batch_size=2, show_loss=False, augmenter=False):

	training_samples = len(loader.image_train_list)
	steps_per_epoch = (training_samples / batch_size) + 1

	for epoch in xrange(epochs):
		print('epoch: '+ str(epoch))
		for step in xrange(steps_per_epoch): # for every batch
			with tf.GradientTape() as g:
				# get batch
				x, y, mask = loader.get_batch(size=batch_size, train=True, augmenter=augmenter)
				y = y[:, :, :, :loader.n_classes] # eliminate the ignore labels channel for computing the loss
				[x, y, mask] = convert_to_tensors([x, y, mask])

				y_ = model(x, training=True)

				loss = tf.losses.softmax_cross_entropy(y, y_, weights=mask)
				if show_loss: print('Training loss: ' + str(loss.numpy()))

			# Gets gradients and applies them
			grads = g.gradient(loss, model.variables)
			optimizer.apply_gradients(zip(grads, model.variables))

		train_acc = get_accuracy(loader, model, train=True)
		test_acc = get_accuracy(loader, model, train=False)
		print('Train accuracy: ' + str(train_acc.numpy()))
		print('Test accuracy: ' + str(test_acc.numpy()))


# Erase the elements if they are from ignore class
def erase_ignore_pixels(labels, predictions):

		indices = tf.squeeze(tf.where(tf.less_equal(labels, loader.n_classes - 1)))  # ignore all labels >= num_classes
		labels = tf.cast(tf.gather(labels, indices), tf.int64)
		predictions = tf.gather(predictions, indices)

		return labels, predictions

# get accuracy from the model
def get_accuracy(loader, model, train=True):
	accuracy = tfe.metrics.Accuracy()
	
	if train: samples = len(loader.image_train_list)
	else: samples = len(loader.image_test_list)

	# Get train accuracy
	for step in xrange(samples): # for every batch
		x, y, mask = loader.get_batch(size=1, train=train, augmenter=False)
		[x, y] = convert_to_tensors([x, y])

		y_ = model(x, training=train)
		
		# Rephape
		y = tf.reshape(y, [y.shape[1] * y.shape[2] * y.shape[0], y.shape[3]])
		y_ = tf.reshape(y_, [y_.shape[1] * y_.shape[2] * y_.shape[0], y_.shape[3]])

		labels, predictions = erase_ignore_pixels(labels=tf.argmax(y, 1), predictions=tf.argmax(y_, 1))

		accuracy(labels, predictions)
		# get the train and test accuracy from the model
	return accuracy.result()

if __name__ == "__main__":

	n_classes = 11
	dataset_path = 'camvid'
	loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation', width=224,
	                height=224, ignore_label=n_classes)
	
	# build model and optimizer
	model = MnasnetEager.MnasnetFC(num_classes=n_classes)

 	# optimizer
	optimizer = tf.train.AdamOptimizer(0.001)

	train(loader=loader, model=model, epochs=100, batch_size=8)
	get_params(model)
	

