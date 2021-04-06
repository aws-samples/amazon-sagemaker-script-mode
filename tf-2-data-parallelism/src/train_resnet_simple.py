import os
import argparse
import numpy as np
import tensorflow as tf
import smdebug.tensorflow as smd

from model_def import get_resnet50

hook = smd.KerasHook.create_from_json_file()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()


def get_train_data(train_dir):
    train_images = np.load(os.path.join(train_dir, 'train_images.npy'))
    train_labels = np.load(os.path.join(train_dir, 'train_labels.npy'))
    print('train_images', train_images.shape, 'train_labels', train_labels.shape)

    return train_images, train_labels


def get_test_data(test_dir):
    test_images = np.load(os.path.join(test_dir, 'test_images.npy'))
    test_labels = np.load(os.path.join(test_dir, 'test_labels.npy'))
    print('test_images', test_images.shape, 'test_labels', test_labels.shape)

    return test_images, test_labels


if __name__ == "__main__":
    tf.random.set_seed(42)
    args, _ = parse_args()

    batch_size = args.batch_size
    n_epochs = args.epochs
    learning_rate = args.learning_rate

    train_images, train_labels = get_train_data(args.train)
    test_images, test_labels = get_test_data(args.test)

    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset = dataset.batch(batch_size)

    tl_model = get_resnet50()

    # Wrap the optimizer with wrap_optimizer so smdebug can find gradients to save
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = hook.wrap_optimizer(optimizer)

    tl_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    hook.set_mode(mode=smd.modes.TRAIN)
    history = tl_model.fit(dataset, epochs=n_epochs,
                           validation_data=(test_images, test_labels),
                           callbacks=[hook]
                           )

    hook.set_mode(mode=smd.modes.EVAL)
    tl_model.evaluate(test_images, test_labels,
                      callbacks=[hook]
                      )
    # save model
    tl_model.save(args.model_dir)
