## Amazon SageMaker Script Mode Examples

This repository contains examples and related resources regarding Amazon SageMaker Script Mode and SageMaker Processing. With Script Mode, you can use training scripts similar to those you would use outside SageMaker with SageMaker's prebuilt containers for various deep learning frameworks such TensorFlow, PyTorch, and Apache MXNet.  Similarly, in SageMaker Processing, you can supply ordinary data preprocessing scripts for almost any language or technology you wish to use, such as the R programming language.  

Currently this repository has the following resources:

- **TensorFlow resources:**  

  - [**TensorFlow 2 Workflow**](tf-2-workflow):  This example shows a complete workflow for TensorFlow 2.  To begin, SageMaker Processing is used to transform the dataset.  Next, Local Mode training and Local Mode endpoints are demonstrated for prototyping training and inference code, respectively.  Automatic Model Tuning is used to automate the hyperparameter tuning process.  Additionally, the AWS Step Functions Data Science SDK is used to automate the project workflow for production-ready environments outside notebooks.  **PREREQUISITES:**  From the *tf-2-workflow* directory, upload ONLY the Jupyter notebook `tf-2-workflow.ipynb`.  

  - [**TensorFlow 2 Sentiment Analysis**](tf-sentiment-script-mode):  SageMaker's prebuilt TensorFlow 2 container is used in this example to train a custom sentiment analysis model. In addition to demonstrating Local Mode training for prototyping your code, this example also shows distributed hosted training in SageMaker with a multi-GPU inance, and usage of SageMaker Batch Transform for asynchronous, large scale inference. **PREREQUISITES:**  From the *tf-sentiment-script-mode* directory, upload ONLY the Jupyter notebook `sentiment-analysis.ipynb`.  

  - [**TensorFlow Distributed Training Options**](tf-distribution-options): This example demonstrates two different distributed training options in SageMaker's Script Mode:  (1) parameter servers, and (2) Horovod. **PREREQUISITES:**  From the *tf-distribution-options* directory, upload ONLY the Jupyter notebook `tf-distributed-training.ipynb`.

  - [**TensorFlow Highly Performant Batch Inference & Training**](tf-batch-inference-script):  The focus of this example is highly performant batch inference using TensorFlow Serving, along with Horovod distributed training. To transform the input image data for inference, a preprocessing script is used with the Amazon SageMaker TensorFlow Serving container.  **PREREQUISITES:**  be sure to upload all files in the *tf-batch-inference-script* directory (including the subdirectory code and files) to the directory where you will run the related Jupyter notebook.  

  - [**TensorFlow Text Classification with Word Embeddings**](tf-word-embeddings): In this example, TensorFlow's tf.keras API is used with Script Mode for a text classification task. An important aspect of the example is showing how to load preexisting word embeddings such as GloVe in Script Mode.  Other features demonstrated include Local Mode endpoints as well as Local Mode training. **PREREQUISITES:**  (1) Use a GPU-based (P3 or P2) SageMaker notebook instance, and (2) be sure to upload all files in the *tf-word-embeddings* directory (including subdirectory *code*) to the directory where you will run the related Jupyter notebook. 

  - [**TensorFlow with Horovod & Inference Pipeline**](tf-horovod-inference-pipeline):  Script Mode with TensorFlow is used for a computer vision task, in a demonstration of Horovod distributed training and doing batch inference in conjunction with an Inference Pipeline for transforming image data before inputting it to the model container. This is an alternative to the previous example, which uses a preprocessing script with the Amazon SageMaker TensorFlow Serving Container rather than an Inference Pipeline. **PREREQUISITES:**  be sure to upload all files in the *tf-horovod-inference-pipeline* directory (including the subdirectory code and files) to the directory where you will run the related Jupyter notebook.  

  - [**TensorFlow Eager Execution**](tf-eager-script-mode):  NOTE:  This example has been superseded by the **TensorFlow 2 Workflow** example above.  This example shows how to use Script Mode with Eager Execution mode in TensorFlow 1.x, a more intuitive and dynamic alternative to the original graph mode of TensorFlow.  It is the default mode of TensorFlow 2.  Local Mode and Automatic Model Tuning also are demonstrated. **PREREQUISITES:**  From the *tf-eager-script-mode* directory, upload ONLY the Jupyter notebook `tf-boston-housing.ipynb`.  


- **R resources:**  

  - [**R in SageMaker Processing**](r-in-sagemaker-processing): In this example, R is used to perform some operations on a dataset and generate a plot within SageMaker Processing.  The job results including the plot image are retrieved and displayed, demonstrating how R can be easily used within a SageMaker workflow. **PREREQUISITES:**  From the *r-in-sagemaker-processing* directory, upload the Jupyter notebook `r-in-sagemaker_processing.ipynb`.
  


- **Miscellaneous resources:**  

  - [**K-means clustering**](k-means-clustering): Most of the samples in this repository involve supervised learning tasks in Amazon SageMaker Script Mode.  For this example, by contrast, we'll undertake an unsupervised learning task, and do so with the Amazon SageMaker K-means built-in algorithm rather than Script Mode.  **PREREQUISITES:**  From the *k-means-clustering* directory, upload the Jupyter notebook `k-means-clustering.ipynb`.
  


## License

The contents of this repository are licensed under the Apache 2.0 License except where otherwise noted. 
