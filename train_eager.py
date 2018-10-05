import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import nets.MnasnetEager as MnasnetEager
import nets.Network as ResnetFCN
import Loader
from sklearn.metrics import confusion_matrix
import math
from utils.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, init_model, erase_ignore_pixels, compute_iou, get_metrics

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(7)
np.random.seed(7)



# Trains the model for certains epochs on a dataset
def train(loader, model, epochs=5, batch_size=2, show_loss=False, augmenter=False, lr=None, init_lr=2e-4):

    training_samples = len(loader.image_train_list)
    steps_per_epoch = (training_samples / batch_size) + 1
    for epoch in xrange(epochs):
        lr_decay(lr, init_lr, 1e-8, epoch, epochs-1)
        print('epoch: ' + str(epoch) + '. Learning rate: ' + str(lr.numpy()) )

        for step in xrange(steps_per_epoch):  # for every batch
            with tf.GradientTape() as g:
                # get batch
                x, y, mask = loader.get_batch(size=batch_size, train=True, augmenter=augmenter)
                x = preprocess(x, mode='imagenet')
                y = y[:, :, :, :loader.n_classes]  # eliminate the ignore labels channel for computing the loss
                [x, y, mask] = convert_to_tensors([x, y, mask]) 

                y_ = model(x, training=True)

                loss = tf.losses.softmax_cross_entropy(y, y_, weights=mask)
                if show_loss: print('Training loss: ' + str(loss.numpy()))

            # Gets gradients and applies them
            grads = g.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

        train_acc, train_miou = get_metrics(loader, model, loader.n_classes,  train=True)
        test_acc_scaled_flp, test_miou_scaled_flp = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True, scales=[ 0.75, 1.5, 1, 2, 0.5])
        test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=False, scales=[1])

        print('Train accuracy: ' + str(train_acc.numpy()))
        print('Train miou: ' + str(train_miou))
        print('Test accuracy: ' + str(test_acc.numpy()))
        print('Test accuracy scaled: ' + str(test_acc_scaled_flp.numpy()))
        print('Test miou: ' + str(test_miou))
        print('Test miou scaled: ' + str(test_miou_scaled_flp))
        print ''


if __name__ == "__main__":
    n_classes = 11
    batch_size = 6
    epochs = 250
    width = 448
    height = 448
    lr = 3e-4
    dataset_path = 'Datasets/camvid'
    loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation', width=width, height=height,median_frequency=0.0 )

    # build model and optimizer
    model = ResnetFCN.ResnetFCN(num_classes=n_classes)

    # optimizer
    learning_rate = tfe.Variable(lr)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Init model (variables and input shape)
    init_model(model, input_shape=(batch_size, width, height, 3))

    # Init saver
    saver_model = tfe.Saver(var_list=model.variables) # can use also ckpt = tfe.Checkpoint((model=model, optimizer=optimizer,learning_rate=learning_rate, global_step=global_step)

    restore_state(saver_model, 'weights/last_saver')

    get_params(model)

    train(loader=loader, model=model, epochs=epochs, batch_size=batch_size, augmenter='segmentation', lr=learning_rate, init_lr=lr)
    saver_model.save('weights/last_saver')





