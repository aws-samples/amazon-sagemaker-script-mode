import argparse
import json
import os

import tensorflow as tf
import horovod.tensorflow.keras as hvd
import smdebug.tensorflow as smd

from utils import get_train_data
from model_def import get_resnet50

hook = smd.KerasHook.create_from_json_file()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, required=False, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, required=False, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--eval', type=str, required=False, default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir', type=str, required=True, help='The directory where the model will be stored.')
    parser.add_argument('--model_output_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--tensorboard-dir', type=str, default=os.environ.get('SM_MODULE_DIR'))
    parser.add_argument('--weight-decay', type=float, default=2e-4, help='Weight decay for convolutions.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data-config', type=json.loads, default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--fw-params', type=json.loads, default=os.environ.get('SM_FRAMEWORK_PARAMS'))
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default='0.9')

    args = parser.parse_args()

    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    scaled_bs = args.batch_size * hvd.size()
    dataset = get_train_data(args.train, scaled_bs)

    model = get_resnet50()

    # Horovod: adjust learning rate based on number of GPUs.
    scaled_lr = args.learning_rate * hvd.size()
    opt = tf.optimizers.Adam(scaled_lr)

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(
        optimizer=opt,
        #         backward_passes_per_step=1,
        #         average_aggregated_gradients=True
    )

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                  optimizer=opt,
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    # Train the model.
    # Horovod: adjust number of steps based on number of GPUs.
    history = model.fit(dataset, steps_per_epoch=500 // hvd.size(),
                        callbacks=callbacks,
                        epochs=args.epochs,
                        verbose=verbose
                        )
    if hvd.rank() == 0:
        #         save_history(args.model_dir + "/hvd_history.p", history)
        model.save(args.model_dir + "/hvd_history.p")
