import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import nets.MnasnetEager as MnasnetEager
import Loader
from sklearn.metrics import confusion_matrix

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
    print("Total parameters of the net: " + str(total_parameters) + " == " + str(total_parameters / 1000000.0) + "M")


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
        print('epoch: ' + str(epoch))
        for step in xrange(steps_per_epoch):  # for every batch
            with tf.GradientTape() as g:
                # get batch
                x, y, mask = loader.get_batch(size=batch_size, train=True, augmenter=augmenter)
                y = y[:, :, :, :loader.n_classes]  # eliminate the ignore labels channel for computing the loss
                [x, y, mask] = convert_to_tensors([x, y, mask])

                y_ = model(x, training=True)

                loss = tf.losses.softmax_cross_entropy(y, y_, weights=mask)
                if show_loss: print('Training loss: ' + str(loss.numpy()))

            # Gets gradients and applies them
            grads = g.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

        train_acc, train_miou = get_metrics(loader, model, train=True)
        test_acc, test_miou = get_metrics(loader, model, train=False)
        print('Train accuracy: ' + str(train_acc.numpy()))
        print('Train miou: ' + str(train_miou))
        print('Test accuracy: ' + str(test_acc.numpy()))
        print('Test miou: ' + str(test_miou))


# Erase the elements if they are from ignore class
def erase_ignore_pixels(labels, predictions, mask):
    indices = tf.squeeze(tf.where(tf.greater(mask, 0)))  # ignore labels
    labels = tf.cast(tf.gather(labels, indices), tf.int64)
    predictions = tf.gather(predictions, indices)

    return labels, predictions


# get accuracy from the model
def get_metrics(loader, model, train=True):
    accuracy = tfe.metrics.Accuracy()
    conf_matrix = np.zeros ((n_classes, n_classes))
    if train:
        samples = len(loader.image_train_list)
    else:
        samples = len(loader.image_test_list)

    # Get train accuracy
    for step in xrange(samples):  # for every batch
        x, y, mask = loader.get_batch(size=1, train=train, augmenter=False)
        [x, y] = convert_to_tensors([x, y])

        y_ = model(x, training=train)

        # Rephape
        y = tf.reshape(y, [y.shape[1] * y.shape[2] * y.shape[0], y.shape[3]])
        y_ = tf.reshape(y_, [y_.shape[1] * y_.shape[2] * y_.shape[0], y_.shape[3]])
        mask = tf.reshape(mask, [mask.shape[1] * mask.shape[2] * mask.shape[0]])

        labels, predictions = erase_ignore_pixels(labels=tf.argmax(y, 1), predictions=tf.argmax(y_, 1), mask=mask)
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



if __name__ == "__main__":
    n_classes = 11
    dataset_path = 'Datasets/camvid'
    loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation', width=128,
                           height=128)

    # build model and optimizer
    model = MnasnetEager.MnasnetFC(num_classes=n_classes)

    # optimizer
    optimizer = tf.train.AdamOptimizer(0.0003)

    train(loader=loader, model=model, epochs=100, batch_size=8)
    get_params(model)
