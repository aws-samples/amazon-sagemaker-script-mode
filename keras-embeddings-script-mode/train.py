import argparse
import os
import sys
import numpy as np
import tensorflow as tf


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


def get_embeddings(embedding_dir):
    
    embeddings = np.load(os.path.join(embedding_dir, 'embedding.npy'))
    print('embeddings shape:  ', embeddings.shape)

    return embeddings


def get_model(embedding_dir, NUM_WORDS, WORD_INDEX_LENGTH, LABELS_INDEX_LENGTH, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
    
    embedding_matrix = get_embeddings(embedding_dir)
    
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = tf.keras.layers.Embedding(NUM_WORDS,
                                                EMBEDDING_DIM,
                                                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                input_length=MAX_SEQUENCE_LENGTH,
                                                trainable=False)
    '''
    
    # initialize embedding layer from scratch and learn its weights
    embedding_layer = Embedding(WORD_INDEX_LENGTH + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH)
    '''

    # train a 1D convnet with global maxpooling
    sequence_input = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = tf.keras.layers.Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    x = tf.keras.layers.Conv1D(128, 5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    x = tf.keras.layers.Conv1D(128, 5, activation='relu')(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    preds = tf.keras.layers.Dense(LABELS_INDEX_LENGTH, activation='softmax')(x)
    
    return tf.keras.Model(sequence_input, preds)
    

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
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)

