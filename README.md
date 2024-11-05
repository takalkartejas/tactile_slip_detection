# Tactile slip detection

This repository contains a programs to train a model to classify cases between slip and non slip based on the provided tactile images and labels. The code contains deep learning training algorithm, machine learning training algorithm and data preprocessing algorithm. The final summary of relavent results is stored in tune log.

## Dependencies
* Ubuntu 20.04
* python 3.10.12
* Refer requirements.txt for the python libraries

## Getting started
* Ensure that above mentioned dependencies are met
* The best way is to use python environment and install the libraries from requirements.txt
* Open remove_blank_images.ipynb inside pre_processing_data folder and change the location of the datadir to the location of the tactile images and run the code
* Open create_train_test_data.ipynb inside pre_processing_data folder and again change the location fo the datadir to the location of tactile images and set the train_data_dir and test_data_dir to the folders where the processed training and testing data will go.
* Deep learning code:- in deep_learning folder open deep_nn.py set the train and test data directories in the manage_data class, set the hyper parameters inside tuning class. Run the code to start training.
* Machine learning code:- Go to machine_learning folder and open the ml_feature_generation.py code, set the test and train data directory as well as the directory where the HOGs features will be saved inside the manage_data class. Run the feature generation code. After that set the features diretory inside the ml_train.py code and the program.


## The file structure
* Deep learning folder contains the deep_learning algorithm, deep_nn.py is the running version
* Machine learning folder containes the machine_learning algorithm, ml_feature_generation.py and ml_train.py are the running versions.
* Pre processing data containes the preprocessing programs for training.
* Tune log contains the relavent log of training along with latest models
     
## Authors
* Tejas Takalkar