import argparse
import os
import sys
import numpy as np
import tensorflow as tf

from model_def import get_model


def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    
    parser.add_argument('--num_words', type=int)
    parser.add_argument('--word_index_len', type=int)
    parser.add_argument('--labels_index_len', type=int)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--max_sequence_len', type=int)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    
    # embedding directory
    parser.add_argument('--embedding', type=str, default=os.environ.get('SM_CHANNEL_EMBEDDING'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()


def get_train_data(train_dir):
    
    x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    print('x train', x_train.shape,'y train', y_train.shape)

    return x_train, y_train


def get_val_data(val_dir):
    
    x_val = np.load(os.path.join(val_dir, 'x_val.npy'))
    y_val = np.load(os.path.join(val_dir, 'y_val.npy'))
    print('x val', x_val.shape,'y val', y_val.shape)

    return x_val, y_val
 

if __name__ == "__main__":
            
    args, _ = parse_args()
    
    x_train, y_train = get_train_data(args.train)
    x_val, y_val = get_val_data(args.val)
    
    model = get_model(args.embedding, 
                      args.num_words,
                      args.word_index_len,
                      args.labels_index_len, 
                      args.embedding_dim, 
                      args.max_sequence_len)
    
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_val, y_val)) 
    
    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    model.save(args.model_dir + '/1')

