## Amazon SageMaker Script Mode Examples

This repository contains examples and related resources regarding Amazon SageMaker’s Script Mode. With Script Mode, you can use training scripts similar to those you would use outside SageMaker with SageMaker's prebuilt containers for various deep learning frameworks such TensorFlow, PyTorch, and Apache MXNet.

Currently this repository has the following resources:

- [**TensorFlow Eager Execution**](tf-eager-script-mode):  This example shows how to use Script Mode with TensorFlow's Eager Execution mode. Eager Execution is the future of TensorFlow, and a major paradigm shift. Introduced as a more intuitive and dynamic alternative to the original graph mode of TensorFlow, Eager Execution will be the default mode of TensorFlow 2. **PREREQUISITES:**  Be sure to upload all four files to the directory where you will run the related Jupyter notebook.  

- [**Keras Text Classification with Word Embeddings**](keras-embeddings-script-mode): In this example, Keras is used with Script Mode for a text classification task. An important aspect of the example is showing how to load preexisting word embeddings such as GloVe in Script Mode. **PREREQUISITES:**  (1) Use a GPU-based (P3 or P2) SageMaker notebook instance, and (2) be sure to upload all files (except the LICNESE) to the directory where you will run the related Jupyter notebook. 

## License

The contents of this repository are licensed under the Apache 2.0 License except where otherwise noted. 
