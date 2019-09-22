## Amazon SageMaker Script Mode Examples

This repository contains examples and related resources regarding Amazon SageMaker’s Script Mode. With Script Mode, you can use training scripts similar to those you would use outside SageMaker with SageMaker's prebuilt containers for various deep learning frameworks such TensorFlow, PyTorch, and Apache MXNet.

Currently this repository has the following resources:

- [**TensorFlow Eager Execution**](tf-eager-script-mode):  This example shows how to use Script Mode with TensorFlow's Eager Execution mode. Eager Execution is the future of TensorFlow, and a major paradigm shift. Introduced as a more intuitive and dynamic alternative to the original graph mode of TensorFlow, Eager Execution is the default mode of TensorFlow 2. **PREREQUISITES:**  From the *tf-eager-script-mode* directory, upload ONLY the Jupyter notebook `tf-boston-housing.ipynb`.  

- [**TensorFlow Sentiment Analysis**](tf-sentiment-script-mode):  Script Mode is used with TensorFlow's tf.keras implementation of the Keras API for a sentiment analysis task. In addition to demonstrating Local Mode training for testing your code, this example also shows usage of SageMaker Batch Transform for asynchronous, large scale inference. **PREREQUISITES:**  From the *tf-sentiment-script-mode* directory, upload ONLY the Jupyter notebook `sentiment-analysis.ipynb`.  

- [**TensorFlow Text Classification with Word Embeddings**](keras-embeddings-script-mode): In this example, TensorFlow's tf.keras API is used with Script Mode for a text classification task. An important aspect of the example is showing how to load preexisting word embeddings such as GloVe in Script Mode.  Other features demonstrated include Local Mode endpoints as well as Local Mode training. **PREREQUISITES:**  (1) Use a GPU-based (P3 or P2) SageMaker notebook instance, and (2) be sure to upload all files in the *keras-embeddings-script-mode* directory (including subdirectory *code*) to the directory where you will run the related Jupyter notebook. 

- [**TensorFlow Distributed Training Options**](tf-distribution-options): This example demonstrates distributed training options in Script Mode, including the use of parameter servers and Horovod. **PREREQUISITES:**  be sure to upload all files in the *tf-distribution-options* directory (including the subdirectory code and files) to the directory where you will run the related Jupyter notebook.

- [**TensorFlow Highly Performant Batch Inference & Training**](tf-batch-inference-script):  The focus of this example is highly performant batch inference using TensorFlow Serving, along with Horovod distributed training. To transform the input image data for inference, a preprocessing script is used with the Amazon SageMaker TensorFlow Serving container.  **PREREQUISITES:**  be sure to upload all files in the *tf-batch-inference-script* directory (including the subdirectory code and files) to the directory where you will run the related Jupyter notebook.  

- [**TensorFlow with Horovod & Inference Pipeline**](tf-horovod-inference-pipeline):  Script Mode with TensorFlow is used for a computer vision task, in a demonstration of Horovod distributed training and doing batch inference in conjunction with an Inference Pipeline for transforming image data before inputting it to the model container. This is an alternative to the previous example, which uses a preprocessing script with the Amazon SageMaker TensorFlow Serving Container rather than an Inference Pipeline. **PREREQUISITES:**  be sure to upload all files in the *tf-horovod-inference-pipeline* directory (including the subdirectory code and files) to the directory where you will run the related Jupyter notebook.  


## License

The contents of this repository are licensed under the Apache 2.0 License except where otherwise noted. 
