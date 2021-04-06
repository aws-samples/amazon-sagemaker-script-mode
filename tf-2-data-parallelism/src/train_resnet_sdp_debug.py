# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import logging
import argparse
import numpy as np

import tensorflow as tf
# Import SMDataParallel TensorFlow2 Modules
import smdistributed.dataparallel.tensorflow as dist

from model_def import get_resnet50

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

tf.random.set_seed(42)
# SMDataParallel: Initialize
dist.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    # SMDataParallel: Pin GPUs to a single SMDataParallel process [use SMDataParallel local_rank() API]
    tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], 'GPU')


def get_train_data(train_dir, batch_size):
    train_images = np.load(os.path.join(train_dir, 'train_images.npy'))
    train_labels = np.load(os.path.join(train_dir, 'train_labels.npy'))
    logger.info('train_images', train_images.shape, 'train_labels', train_labels.shape)

    dataset_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset_train = dataset_train.repeat().shuffle(10000).batch(batch_size)

    return dataset_train


def train(args):
    # Load data from S3
    #     train_dir = os.environ.get('SM_CHANNEL_TRAIN')
    train_dir = args.train
    batch_size = args.batch_size
    dataset = get_train_data(train_dir, batch_size)

    model = get_resnet50(transfer_learning=True)

    loss_fn = tf.losses.SparseCategoricalCrossentropy()
    acc = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # SMDataParallel: dist.size()
    # LR for 8 node run : 0.000125
    # LR for single node run : 0.001
    opt = tf.optimizers.Adam(args.learning_rate * dist.size())

    checkpoint_dir = os.environ['SM_MODEL_DIR']
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)

    @tf.function
    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = model(images, training=True)
            loss_value = loss_fn(labels, probs)
            acc_value = acc(labels, probs)

        # SMDataParallel: Wrap tf.GradientTape with SMDataParallel's DistributedGradientTape
        tape = dist.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if first_batch:
            # SMDataParallel: Broadcast model and optimizer variables
            dist.broadcast_variables(model.variables, root_rank=0)
            dist.broadcast_variables(opt.variables(), root_rank=0)

        # SMDataParallel: all_reduce call
        loss_value = dist.oob_allreduce(loss_value)  # Average the loss across workers
        acc_value = dist.oob_allreduce(acc_value)

        return loss_value, acc_value

    for epoch in range(args.epochs):
        for batch, (images, labels) in enumerate(dataset.take(10000 // dist.size())):
            loss_value, acc_value = training_step(images, labels, batch == 0)

            if batch % 100 == 0 and dist.rank() == 0:
                logger.info(
                    '*** Epoch %d   Step   #%d Accuracy: %.6f   Loss: %.6f ***' % (epoch, batch, acc_value, loss_value))

    # SMDataParallel: Save checkpoints only from master node.
    if dist.rank() == 0:
        model.save(os.path.join(checkpoint_dir, '1'))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)