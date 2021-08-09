#!/usr/bin/env python

import numpy as np
import os
import pandas as pd 
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__=='__main__':
    
    install('torch')
    install('transformers')
    install('datasets[s3]')
    
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # tokenizer used in preprocessing
    tokenizer_name = 'distilbert-base-uncased'

    # dataset used
    dataset_name = 'imdb'

    # load dataset
    dataset = load_dataset(dataset_name)

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    # load dataset
    train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
    test_dataset = test_dataset.shuffle().select(range(1000)) # smaller the size for test dataset to 1k 

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set format for pytorch
    train_dataset =  train_dataset.rename_column("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_dataset.save_to_disk('/opt/ml/processing/train')
    test_dataset.save_to_disk('/opt/ml/processing/test')