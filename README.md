## Amazon SageMaker Script Mode Examples

This repository contains examples and related resources regarding Amazon SageMaker Script Mode and SageMaker Processing. With Script Mode, you can use training scripts similar to those you would use outside SageMaker with SageMaker's prebuilt containers for various frameworks such TensorFlow and PyTorch.  Similarly, in SageMaker Processing, you can supply ordinary data preprocessing scripts for almost any language or technology you wish to use, such as the R programming language.  

Currently this repository has resources for **Hugging Face**, **TensorFlow**, **Bring Your Own** (BYO models, plus Script Mode-style experience with your own containers), and **Miscellaneous** (Script Mode-style experience for SageMaker Processing etc.). There also is an **Older Resources** section with examples of older framework versions.  

For those new to SageMaker, there is a set of 2-hour workshops covering the basics at [**Amazon SageMaker Workshops**](https://github.com/awslabs/amazon-sagemaker-workshop).

- **Hugging Face Resources:**

  - [**Hugging Face automated model training and deployment in SageMaker Pipelines**](hugging-face-lambda-step):  This example uses the SageMaker prebuilt Hugging Face (PyTorch) container in an end-to-end demo with model training and deployment within SageMaker Pipelines.  A lightweight model deployment is performed by a SageMaker Pipeline Lambda step.  **PREREQUISITES:**  either clone this repository, or from the *hugging-face-lambda-step* directory, upload all files and folders; then run the notebook `sm-pipelines-hugging-face-lambda-step.ipynb`.

- **TensorFlow Resources:**  

  - [**TensorFlow 2 Sentiment Analysis**](tf-sentiment-script-mode):  SageMaker's prebuilt TensorFlow 2 container is used in this example to train a custom sentiment analysis model. Distributed hosted training in SageMaker is performed on a multi-GPU instance, using the native TensorFlow `MirroredStrategy`.  Additionally, SageMaker Batch Transform is used for asynchronous, large scale inference/batch scoring. **PREREQUISITES:**  From the *tf-sentiment-script-mode* directory, upload ONLY the Jupyter notebook `sentiment-analysis.ipynb`.  

  - [**TensorFlow 2 Workflow with SageMaker Pipelines**](tf-2-workflow-smpipelines):  This example shows a complete workflow for TensorFlow 2, starting with prototyping followed by automation with [Amazon SageMaker Pipelines](https://aws.amazon.com/sagemaker/pipelines).  To begin, SageMaker Processing is used to transform the dataset.  Next, Local Mode training and Local Mode endpoints are demonstrated for prototyping training and inference code, respectively.  Automatic Model Tuning is used to automate the hyperparameter tuning process.  Finally, the workflow is automated with SageMaker Pipelines.  **PREREQUISITES:** If you wish to run the Local Mode sections of the example, use a SageMaker Notebook Instance rather than SageMaker Studio.  From the *tf-2-workflow-smpipelines* directory, upload ONLY the Jupyter notebook `tf-2-workflow-smpipelines.ipynb`.  
  
  - [**TensorFlow 2 with SageMaker Distributed Data Parallel Library**](tf-2-data-parallelism):  This example utilizes SageMaker's distributed data parallel library, which extends SageMaker’s training capabilities on deep learning models with near-linear scaling efficiency, achieving fast time-to-train with minimal code changes.  It is applied to a TensorFlow 2 image classification model trained on the CIFAR10 dataset.  **PREREQUISITES:**  Be sure your AWS account limits allow usage of two ml.p3.16xlarge instances.  
  
  - [**TensorFlow 2 Loading Pretrained Embeddings for Classification Tasks**](tf-2-word-embeddings): In this example, TensorFlow 2 is used with Script Mode for a text classification task. An important aspect of the example is showing how to load pretrained embeddings in Script Mode. This illustrates one aspect of the flexibility of SageMaker Script Mode for setting up training jobs: in addition to data, you can pass in arbitrary files needed for training (not just embeddings).  **PREREQUISITES:**  (1) be sure to upload all files in the *tf-2-word-embeddings* directory (including subdirectory *code*) to the directory where you will run the related Jupyter notebook.
  
  
- **Bring Your Own (BYO) Resources:**  

  - [**lightGBM BYO**](lightgbm-byo): In this repository, most samples use Amazon SageMaker prebuilt framework containers for TensorFlow and other frameworks.  For this example, however, we'll show how to BYO container to create a Script Mode-style experience similar to a prebuilt SageMaker framework container, using lightGBM, a popular gradient boosting framework.  **PREREQUISITES:**  From the *lightgbm-byo* directory, upload the Jupyter notebook `lightgbm-byo.ipynb`.

  - [**Deploy Pretrained Models**](deploy-pretrained-model):  In addition to the ease of use of the SageMaker Python SDK for model training in Script Mode, the SDK also enables you to easily BYO model.  In this example, the SageMaker prebuilt PyTorch container is used to demonstrate how you can quickly take a pretrained or locally trained model and deploy it in a SageMaker hosted endpoint. There are examples for both OpenAI's GPT-2 and BERT. **PREREQUISITES:**  From the *deploy-pretrained-model* directory, upload the entire BERT or GPT2 folder's contents, depending on which model you select. Run either `Deploy_BERT.pynb` or `Deploy_GPT2.ipynb`.  


- **Miscellaneous Resources:**  

  - [**R in SageMaker Processing**](r-in-sagemaker-processing): SageMaker Script Mode is directed toward making the model training process easier.  However, an experience similar to Script Mode also is available for SageMaker Processing:  you can bring in your data processing scripts and easily run them on managed infrastructure either with BYO containers or prebuilt containers for frameworks such as Spark and Scikit-learn.  In this example, R is used to perform operations on a dataset and generate a plot within SageMaker Processing.  The job results including the plot image are retrieved and displayed, demonstrating how R can be easily used within a SageMaker workflow. **PREREQUISITES:**  From the *r-in-sagemaker-processing* directory, upload the Jupyter notebook `r-in-sagemaker_processing.ipynb`.

  - [**K-means clustering**](k-means-clustering): Most of the samples in this repository involve supervised learning tasks in Amazon SageMaker Script Mode.  For this example, by contrast, we'll undertake an unsupervised learning task, and do so with the Amazon SageMaker K-means built-in algorithm rather than Script Mode.  The SageMaker built-in algorithms were developed for large-scale training tasks and may offer a simpler user experience depending on the use case.  **PREREQUISITES:**  From the *k-means-clustering* directory, upload the Jupyter notebook `k-means-clustering.ipynb`.


- **Older Resources:**  

  - [**TensorFlow 2 Workflow with the AWS Step Functions Data Science SDK**](tf-2-workflow):  **NOTE**:  This example has been superseded by the **TensorFlow 2 Workflow with SageMaker Pipelines** example above. This example shows a complete workflow for TensorFlow 2 with automation by the AWS Step Functions Data Science SDK, an older alternative to [Amazon SageMaker Pipelines](https://aws.amazon.com/sagemaker/pipelines).  To begin, SageMaker Processing is used to transform the dataset.  Next, Local Mode training and Local Mode endpoints are demonstrated for prototyping training and inference code, respectively.  Automatic Model Tuning is used to automate the hyperparameter tuning process.  **PREREQUISITES:**  From the *tf-2-workflow* directory, upload ONLY the Jupyter notebook `tf-2-workflow.ipynb`.  
  
  - [**TensorFlow 1.x (tf.keras) Highly Performant Batch Inference & Training**](tf-batch-inference-script):  The focus of this example is highly performant batch inference using TensorFlow Serving, along with Horovod distributed training. To transform the input image data for inference, a preprocessing script is used with the Amazon SageMaker TensorFlow Serving container.  **PREREQUISITES:**  be sure to upload all files in the *tf-batch-inference-script* directory (including the subdirectory code and files) to the directory where you will run the related Jupyter notebook.  

  - [**TensorFlow 1.x (tf.keras) with Horovod & Inference Pipeline**](tf-horovod-inference-pipeline):  Script Mode with TensorFlow is used for a computer vision task, in a demonstration of Horovod distributed training and doing batch inference in conjunction with an Inference Pipeline for transforming image data before inputting it to the model container. This is an alternative to the previous example, which uses a preprocessing script with the Amazon SageMaker TensorFlow Serving Container rather than an Inference Pipeline. **PREREQUISITES:**  be sure to upload all files in the *tf-horovod-inference-pipeline* directory (including the subdirectory code and files) to the directory where you will run the related Jupyter notebook.  

  
  - [**TensorFlow 1.x (tf.keras) Distributed Training Options**](tf-distribution-options): **NOTE**:  Besides the options listed here for TensorFlow 1.x, there are additional options for TensorFlow 2, including [A] built-in [**SageMaker Distributed Training**](https://aws.amazon.com/sagemaker/distributed-training/) for both data and model parallelism, and [B] native distribution strategies such as MirroredStrategy as demonstrated in the **TensorFlow 2 Sentiment Analysis** example above. This TensorFlow 1.x example demonstrates two other distributed training options for SageMaker's Script Mode:  (1) parameter servers, and (2) Horovod. **PREREQUISITES:**  From the *tf-distribution-options* directory, upload ONLY the Jupyter notebook `tf-distributed-training.ipynb`.

  - [**TensorFlow 1.x (tf.keras) Eager Execution**](tf-eager-script-mode):  **NOTE**:  This TensorFlow 1.x example has been superseded by the **TensorFlow 2 Workflow** example above.  This example shows how to use Script Mode with Eager Execution mode in TensorFlow 1.x, a more intuitive and dynamic alternative to the original graph mode of TensorFlow.  It is the default mode of TensorFlow 2.  Local Mode and Automatic Model Tuning also are demonstrated. **PREREQUISITES:**  From the *tf-eager-script-mode* directory, upload ONLY the Jupyter notebook `tf-boston-housing.ipynb`.  


  
## License

The contents of this repository are licensed under the Apache 2.0 License except where otherwise noted.
