import numpy as np
import os
import tensorflow as tf


def get_embeddings(embedding_dir):
    
    embeddings = np.load(os.path.join(embedding_dir, 'embedding.npy'))
    print('embeddings shape:  ', embeddings.shape)

    return embeddings


def get_model(embedding_dir, NUM_WORDS, WORD_INDEX_LENGTH, LABELS_INDEX_LENGTH, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
    
    embedding_matrix = get_embeddings(embedding_dir)
    
    # trainable = False to keep the embeddings frozen
    embedding_layer = tf.keras.layers.Embedding(NUM_WORDS,
                                                EMBEDDING_DIM,
                                                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                input_length=MAX_SEQUENCE_LENGTH,
                                                trainable=False)

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
   
