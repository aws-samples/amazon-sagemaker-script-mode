## Amazon SageMaker Script Mode Examples

This repository contains examples and related resources regarding Amazon SageMaker’s Script Mode. With Script Mode, you can use training scripts similar to those you would use outside SageMaker with SageMaker's prebuilt containers for various deep learning frameworks such TensorFlow, PyTorch, and Apache MXNet.

Currently this repository has the following resources:

- [**TensorFlow Eager Execution**](tf-eager-script-mode):  This example shows how to use Script Mode with TensorFlow's Eager Execution mode. Eager Execution is the future of TensorFlow, and a major paradigm shift. Introduced as a more intuitive and dynamic alternative to the original graph mode of TensorFlow, Eager Execution is the default mode of TensorFlow 2. **PREREQUISITES:**  Be sure to upload all files in the *tf-eager-script-mode* directory (including the subdirectory *train_model*) to the directory where you will run the related Jupyter notebook.  

- [**TensorFlow Sentiment Analysis**](tf-sentiment-script-mode):  Script Mode is used with TensorFlow's implementation of the Keras API for a sentiment analysis task. In addition to demonstrating Local Mode training for testing your code, this example also shows usage of SageMaker Batch Transform for asynchronous, large scale inference, rather than SageMaker Hosted Endpoints for near real-time inference. **PREREQUISITES:**  Be sure to upload all files in the *tf-sentiment-script-mode* directory to the directory where you will run the related Jupyter notebook.  

- [**Keras Text Classification with Word Embeddings**](keras-embeddings-script-mode): In this example, Keras is used with Script Mode for a text classification task. An important aspect of the example is showing how to load preexisting word embeddings such as GloVe in Script Mode. **PREREQUISITES:**  (1) Use a GPU-based (P3 or P2) SageMaker notebook instance, and (2) be sure to upload all files in the *keras-embeddings-script-mode* directory (including subdirectory *code*) to the directory where you will run the related Jupyter notebook. 

- [**TensorFlow Distributed Training & Inference with Preprocessing Script**](tf-horovod-preprocessing):  In connection with a computer vision task, Script Mode with TensorFlow is used along with Horovod distributed training and real time and batch inference.  To transform the input image data for inference, a preprocessing script is used with the Amazon SageMaker TensorFlow Serving Container.  **PREREQUISITES:**  (1) Run this example **ONLY** in the AWS Regions Ireland, N. Virginia, Ohio, or Oregon; (2) be sure to upload all files in the *tf-horovod-preprocessing* directory (including the subdirectory code and files) to the directory where you will run the related Jupyter notebook.  

- [**TensorFlow Distributed Training & Inference Pipeline**](tf-horovod-inference-pipeline):  Script Mode with TensorFlow is used for a computer vision task, in a demonstration of Horovod distributed training and doing batch inference in conjunction with an Inference Pipeline for transforming image data before inputting it to the model container. This is an alternative to the previous example, which uses a preprocessing script with the Amazon SageMaker TensorFlow Serving Container rather than an Inference Pipeline. **PREREQUISITES:**  (1) Run this example **ONLY** in the Oregon (us-west-2) AWS Region; (2) be sure to upload all files in the *tf-horovod-inference-pipeline* directory (including the subdirectory code and files) to the directory where you will run the related Jupyter notebook.  


## License

The contents of this repository are licensed under the Apache 2.0 License except where otherwise noted. 
