# Functioning

This project is a realization of the recognition of handwritten numbers based on convolutional neural networks (CNN). In this project, the stochastic gradient descent (SGD) is the algorithm mainly used, which is also compared with several other algorithms such as the momentum, AdaGrad and RMSProp. In this project, all functionings are realized by Python and Numpy, without other Python packages like TensorFlow or PyTorch.



# Files

You can find all related programming files of this project in the folder **Projet_CNN**, which contains totally 4 data files, 6 `.ipynb` files and 6 folders.

- **Data files**
  - **train-images.idx3-ubyte** : File of the training data (images of handwritten numbers).
  - **train-labels.idx1-ubyte** : File of the training labels (actual numbers related to the pictures of the training data).
  - **t10k-images.idx3-ubyte** : File of the testing data (images of handwritten numbers).
  - **t10k-labels.idx1-ubyte** : File of the testing labels (actual numbers related to the pictures of the testing data).
- **Model files** (`.ipynb`)
  - **1_Model_sgd_984.ipynb** : Model of CNN with the optimization algorithm **SGD**.
  - **2_Model_sgd_2conv_986.ipynb** : Model of CNN with the optimization algorithm **SGD** where there are two convolutional layers.
  - **3_Model_sgdmd_955.ipynb** : Model of CNN with the optimization algorithm **SGD with momentum and decay of learning rate**.
  - **4_Model_sgdmd_shuffle_956.ipynb** : Model of CNN with the optimization algorithm **SGD with momentum and decay of learning rate** where the training data are shuffled for each epoch.
  - **5_Model_adagrad_971.ipynb** : Model of CNN with the optimization algorithm **AdaGrad**.
  - **6_Model_rmsprop_977.ipynb** : Model of CNN with the optimization algorithm **RMSProp**.

- **Saved model parameter folders**
  - **200612094306** : Saved parameters related to the model file **1_Model_sgd_984.ipynb**.
  - **200612115604** : Saved parameters related to the model file **2_Model_sgd_2conv_986.ipynb**.
  - **200613183131** : Saved parameters related to the model file **3_Model_sgdmd_955.ipynb**.
  - **200614145056** : Saved parameters related to the model file **4_Model_sgdmd_shuffle_956.ipynb**.
  - **200613003817** : Saved parameters related to the model file **5_Model_adagrad_971.ipynb**.
  - **200614141913** : Saved parameters related to the model file **6_Model_rmsprop_977.ipynb**.



# Structure of the model files

The structure of each model file (`.ipynb`) will be presented as follows:

- **Realization of layers in the CNN**
  - **Fully-connected layer** : Realization of forward and backward propagation of a **fully-connected layer**.
  - **Convolutional layer** : Realization of forward and backward propagation of a **convolutional layer**.
  - **Pooling layer - Max pooling** : Realization of forward and backward propagation of a **max-pooling layer**.
  - **Flatten layer** : Realization of forward and backward propagation of a **flatten layer**.
  - **Activate function** : Realization of forward and backward propagation of an **activate function**.
  - **Loss function - mean squared loss, softmax & cross entropy** : Realization of the **loss functions**.
- **Realization of the optimizer**
  
  - **Optimizer** : Realization of the **optimization algorithm**.
  
- **Data processing** 
  - **Load data** : Get data from the data files.
  - **Preprocess data** : Preprocess the data in order to fit the model.

- **Modeling**

  - **Method of evaluation** : Definition of the functions calculating the **accuracy** of the model.

  - **Save model and load model** : Definition of the methods of saving and loading parameters of the model.

  - **Run model** : Definition of the process to train the model.

    ( **ATTENTION** : The results of the model have been printed under this cell. So **DO NOT** run this cell if you don't want to wait for hours and even days to get the results ! )

  - **Test model** : Definition of the methods of testing the model.

    ( **ATTENTION** : The results of the test have been printed under this cell. So **DO NOT** run this cell if you don't want to wait for minutes to get the results ! )

- **Application - Recognition of an image**
  
  - **Application of the model** : Definition of methods to recognize a random image in the testing data to applicate the model.



# Application of the project

- You can verify all codes in a model file (`.ipynb`) by following the structure of each file mentioned above.
- **ATTENTION** : The results of the model have been printed under the cell **Run model**. So **DO NOT** run this cell if you don't want to wait for hours and even days to get the results ! 
- If you want to verify the performance of the model by testing the prediction of a random image, you **ONLY** need to run **the two cells** in the part **Application of the model** which is at the end of each model file.

