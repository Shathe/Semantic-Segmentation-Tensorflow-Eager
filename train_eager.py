import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import nets.Network as Segception
import utils.Loader as Loader
from utils.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, init_model, get_metrics

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(7)
np.random.seed(7)


# Trains the model for certains epochs on a dataset
def train(loader, model, epochs=5, batch_size=2, show_loss=False, augmenter=False, lr=None, init_lr=2e-4, saver=None):
    training_samples = len(loader.image_train_list)
    steps_per_epoch = (training_samples / batch_size) + 1
    best_miou = 0

    for epoch in range(epochs):  # for each epoch
        lr_decay(lr, init_lr, 1e-8, epoch, epochs - 1)  # compute the new lr
        print('epoch: ' + str(epoch) + '. Learning rate: ' + str(lr.numpy()))
        for step in range(steps_per_epoch):  # for every batch
            with tf.GradientTape() as g:
                # get batch
                x, y, mask = loader.get_batch(size=batch_size, train=True, augmenter=augmenter)
                x = preprocess(x, mode='imagenet')
                [x, y, mask] = convert_to_tensors([x, y, mask])

                y_ = model(x, training=True)  # get output of the model

                loss = tf.losses.softmax_cross_entropy(y, y_, weights=mask)  # compute loss
                if show_loss: print('Training loss: ' + str(loss.numpy()))

            # Gets gradients and applies them
            grads = g.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

        # get metrics
        train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True)
        test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=False,
                                          scales=[1])

        print('Train accuracy: ' + str(train_acc.numpy()))
        print('Train miou: ' + str(train_miou))
        print('Test accuracy: ' + str(test_acc.numpy()))
        print('Test miou: ' + str(test_miou))
        print('')

        # save model if bet
        if test_miou > best_miou:
            best_miou = test_miou
            saver_model.save('weights/best2')

        loader.suffle_segmentation()  # sheffle trainign set


if __name__ == "__main__":
    # some parameters
    n_gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)
    n_classes = 11
    batch_size = 2
    epochs = 200
    width = 448
    height = 448
    lr = 2e-4

    dataset_path = 'Datasets/camvid'
    loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation',
                           width=width, height=height, median_frequency=0.0)

    # build model and optimizer
    model = Segception.Segception(num_classes=n_classes)

    # optimizer
    learning_rate = tfe.Variable(lr)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Init model (variables and input shape)
    init_model(model, input_shape=(batch_size, width, height, 3))

    # Init saver
    saver_model = tfe.Saver(
        var_list=model.variables)  # can use also ckpt = tfe.Checkpoint((model=model, optimizer=optimizer,learning_rate=learning_rate, global_step=global_step)

    restore_state(saver_model, 'weights/best2')

    get_params(model)

    train(loader=loader, model=model, epochs=epochs, batch_size=batch_size, augmenter='segmentation', lr=learning_rate,
          init_lr=lr, saver=saver_model)
    saver_model.save('weights/last_saver2')
