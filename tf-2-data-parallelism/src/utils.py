import os
import numpy as np
import tensorflow as tf


def get_train_data(train_dir, batch_size):
    train_images = np.load(os.path.join(train_dir, 'train_images.npy'))
    train_labels = np.load(os.path.join(train_dir, 'train_labels.npy'))
    print('train_images', train_images.shape, 'train_labels', train_labels.shape)

    dataset_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset_train = dataset_train.repeat().shuffle(10000).batch(batch_size)

    return dataset_train


def get_val_data(val_dir):
    test_images = np.load(os.path.join(val_dir, 'validation_images.npy'))
    test_labels = np.load(os.path.join(val_dir, 'validation_labels.npy'))
    print('validation_images', test_images.shape, 'validation_labels', test_labels.shape)

    dataset_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    return dataset_test
