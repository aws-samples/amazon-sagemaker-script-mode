#!/usr/bin/env python

"""Evaluation script for measuring mean squared error."""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import json
import logging
import pathlib
import pickle
import tarfile
import os

import numpy as np
import pandas as pd

from transformers import AutoModelForSequenceClassification, Trainer
from datasets import load_from_disk

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="./hf_model")

    logger.debug(os.listdir('./hf_model'))
    
#     test_dir = "/opt/ml/processing/test/"
#     test_dataset = load_from_disk(test_dir)
    
#     model = AutoModelForSequenceClassification.from_pretrained('./hf_model')

#     trainer = Trainer(model=model)
    
#     eval_result = trainer.evaluate(eval_dataset=test_dataset)

    with open('./hf_model/evaluation.json') as f:
        eval_result = json.load(f)
    
    logger.debug(eval_result)
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(eval_result))
