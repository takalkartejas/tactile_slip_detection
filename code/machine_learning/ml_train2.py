import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# Set the environment variable to use only the GPU with ID 1 (GTX 1080 Ti)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import PIL
import tensorflow as tf
import numpy as np
import os
from glob import glob
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
import pandas as pd
import numpy as np
import pathlib
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
import datetime
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score

import logging
from collections import deque
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import SGDClassifier
len_dataLoader = 0
from sklearn.impute import SimpleImputer
import time
class Manage_data():
    def __init__(self):
        data_dir='/home/rag-tt/workspace/'
        self.train_data_dir = os.path.join(data_dir,'train_data')
        self.test_data_dir = os.path.join(data_dir,'test_data')
        self.hog_features_dir = os.path.join(data_dir,'hog_features3')
        self.train_features_dir = os.path.join(self.hog_features_dir,'train_features')
        self.test_features_dir = os.path.join(self.hog_features_dir,'test_features')
        self.data_dir= pathlib.Path(data_dir)
        
    def count_subdirectories(self,directory):
        try:
            # List all entries in the directory
            entries = os.listdir(directory)
            
            # Filter out the subdirectories
            subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
            
            # Count the subdirectories
            num_subdirectories = len(subdirectories)
            
            print(f"The directory '{directory}' contains {num_subdirectories} subdirectories.")
            return num_subdirectories
            
        except FileNotFoundError:
            print(f"The directory '{directory}' does not exist.")
        except PermissionError:
            print(f"Permission denied: Unable to access '{directory}'.")
        except Exception as e:
            print(f"Error: {e}")
    
    def find_no_of_images(self, data_dir,obj_id):
        image_dir = os.path.join(data_dir, str(obj_id))
        image_dir= pathlib.Path(image_dir)
        no_of_images= len(list(image_dir.glob('*.jpg')))
        return no_of_images
    
    def trim_data(self, label, image_paths, csv_path):
        # Ensure label and image_paths are numpy arrays
        label = np.array(label)
        image_paths = np.array(image_paths)
        
        # Count the total number of zeroes in label
        total_zeroes = np.sum(label == 0)
        
        # Determine the number of zeroes to remove
        zeroes_to_remove = max(0, total_zeroes - tune.no_of_nonslip_data)
        
        # Indices of zero elements
        zero_indices = np.where(label == 0)[0]
        
        # Indices to keep (last self.no_of_nonslip_data zeroes and all ones)
        indices_to_keep = np.concatenate((zero_indices[-tune.no_of_nonslip_data:], np.where(label != 0)[0]))
        indices_to_keep = np.unique(indices_to_keep)
        indices_to_keep = np.sort(indices_to_keep)
        
        # Create the resulting label array
        label_with_few_zeroes = label[indices_to_keep]
        
        # Remove the same number of elements from the start of image_paths
        paths_with_few_zeroes = image_paths[zeroes_to_remove:]
        
        
        min_tranistion_trim_value = tune.min_trim_value
        max_tranistion_trim_value = tune.slip_instant_labels
        trimmed_labels = []
        trimmed_paths = []
        slip_values = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=2, dtype=None, encoding=None)
        i = 0
        for slip_value in slip_values:
            if slip_value < tune.max_labels:
                if max_tranistion_trim_value < slip_value or slip_value < min_tranistion_trim_value:
                    trimmed_labels.append(label_with_few_zeroes[i])
                    trimmed_paths.append(paths_with_few_zeroes[i])
            i += 1
        trimmed_labels = np.array(trimmed_labels)
        trimmed_paths = np.array(trimmed_paths)
        # print(slip_values)
        # print('label_with_few_zeroes =', label_with_few_zeroes)
        # print('paths_with_few_zeroes=', paths_with_few_zeroes.shape)
        # print('trimmed_labels=', trimmed_labels)
        # print('trimed_paths=', trimmed_paths.shape)
        return trimmed_labels, trimmed_paths
    
    def check_pattern(self,label):
        # Ensure arr is a numpy array
        label = np.array(label)
        
        # Find the first occurrence of 1
        first_one_index = np.argmax(label == 1)
        
        if np.all(label == 0):  # If there's no 1 in the array, ensure all are 0
            return
        
        # Check if there's no 1 in the array
        if np.max(label) == 0:
            assert np.all(label == 0), "Array does not follow the pattern: continuous zeroes followed by continuous ones"
            return
        
        # Assert all elements before first_one_index are 0
        assert np.all(label[:first_one_index] == 0), "Array does not follow the pattern: continuous zeroes followed by continuous ones"
        
        # Assert all elements from first_one_index to the end are 1
        assert np.all(label[first_one_index:] == 1), "Array does not follow the pattern: continuous zeroes followed by continuous ones"
    
    def create_slip_instant_labels(self, csv_path):
        label = []
        slip_values = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=2, dtype=None, encoding=None)
        for slip_value in slip_values:
            if slip_value < tune.slip_instant_labels:
                label.append(0)
            else:
                label.append(1)
        return label
 
    def duplicate_n_balance_data(self, labels, image_paths):
        # Convert labels to numpy array for easier manipulation
        labels = np.array(labels)
        image_paths = np.array(image_paths)

        # Get indices of each class
        class_0_indices = np.where(labels == 0)[0]
        class_1_indices = np.where(labels == 1)[0]
            # Check if either class is empty
        if len(class_0_indices) == 0 or len(class_1_indices) == 0:
            # print(f"Skipping balancing for {labels} as one of the classes is missing")
            return labels, image_paths
    
        # Calculate the difference in the number of samples
        diff = len(class_0_indices) - len(class_1_indices)

        if diff > 0:  # More 0s than 1s
            # Randomly duplicate class 1 samples to balance the dataset
            additional_indices = np.random.choice(class_1_indices, size=diff, replace=True)
            labels = np.concatenate([labels, labels[additional_indices]])
            image_paths = np.concatenate([image_paths, image_paths[additional_indices]])
        elif diff < 0:  # More 1s than 0s
            # Randomly duplicate class 0 samples to balance the dataset
            additional_indices = np.random.choice(class_0_indices, size=-diff, replace=True)
            labels = np.concatenate([labels, labels[additional_indices]])
            image_paths = np.concatenate([image_paths, image_paths[additional_indices]])
        # # Shuffle the dataset to mix the duplicated samples
        # shuffle_indices = np.arange(len(labels))
        # np.random.shuffle(shuffle_indices)
        # labels = labels[shuffle_indices]
        # image_paths = image_paths[shuffle_indices]

        return labels, image_paths           
            
    def balance_n_separate_data(self, labels, image_paths):
        # Convert labels to numpy array for easier manipulation
        labels = np.array(labels)
        image_paths = np.array(image_paths)

        # Get indices of each class
        class_0_indices = np.where(labels == 0)[0]
        class_1_indices = np.where(labels == 1)[0]

        labels_0=[]
        labels_1=[]
        image_paths_0=[]
        image_paths_1=[]
        
        # Check if either class is empty
        if len(class_0_indices) == 0 or len(class_1_indices) == 0:
            print(f"Skipping balancing for {labels} as one of the classes is missing")
            return labels_0,labels_1,image_paths_0,image_paths_1


        #separate_data
        for i in class_0_indices:
            labels_0.append(labels[i])
            image_paths_0.append(image_paths[i])
        for i in class_1_indices:
            labels_1.append(labels[i])
            image_paths_1.append(image_paths[i])   
        
        # print('labels_0',labels_0)
        # print('image_paths_0',image_paths_0)
        # print('labels_1',labels_1)
        # print('image_paths_1',image_paths_1)
                 
        # Calculate the difference in the number of samples
        diff = len(class_0_indices) - len(class_1_indices)

        if diff < 0:  # More 1s than 0s
            # Randomly duplicate class 0 samples to balance the dataset
            additional_indices = np.random.choice(class_0_indices, size=-diff, replace=True)
            labels_0 = np.concatenate([labels_0, labels[additional_indices]])
            image_paths_0 = np.concatenate([image_paths_0, image_paths[additional_indices]])
        
        # print('labels_0',labels_0)
        # print('image_paths_0',image_paths_0)
        # print('labels_1',labels_1)
        # print('image_paths_1',image_paths_1)
        
        return labels_0, image_paths_0, labels_1, image_paths_1  
    
    def save_hog_features_and_labels(self, features, labels, obj_id, feature_dir):
        os.makedirs(feature_dir, exist_ok=True)
        labels_file = os.path.join(feature_dir, f'labels_{obj_id}.npy')
        features_file = os.path.join(feature_dir, f'features_{obj_id}.npy')
        
        np.save(labels_file, labels)
        np.save(features_file, features, allow_pickle=True)
        
    def process_features(self,data_dir, no_of_samples, feature_dir):
        file_paths = []
        features = []
        image_paths = []
        labels = []
        
        for obj_id in range(no_of_samples):
            image_paths = []
            no_of_images = self.find_no_of_images(data_dir,obj_id)
            csv_path = os.path.join(data_dir, str(obj_id),'slip_log.csv')
            if no_of_images < 40 or not os.path.exists(csv_path):
                continue
            label = self.create_slip_instant_labels(csv_path)
            
            label2 = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=1, dtype=None, encoding=None)
            
            for img_id in range(no_of_images):
                image_path = os.path.join(data_dir, str(obj_id),'depth'+ str(img_id)+ '.png')
                image_paths.append(image_path)
            self.check_pattern(label)
            
            label, image_paths = self.trim_data(label,image_paths,csv_path)  

            #check if the labels are correct
            np_label = np.array(label)  
            if np.all(np_label==1):
                image_paths = []
                continue 
            
            class_0_indices = np.where(label == 0)[0]
            class_1_indices = np.where(label == 1)[0]
            
            if len(class_0_indices) > len(class_1_indices):
                image_paths = []
                continue
            
            '''            
            example values of label and image_paths pair             
            label size = 83, imagepaths_size = 83 
            zeroes = 41, ones = 42
            img40 == 0, img41 == 1
            '''
            label_0, image_paths_0, label_1, image_paths_1 = self.balance_n_separate_data(label,image_paths) 

            label_0, features_0, label_1, features_1 = ml.create_ml_features(label_0, image_paths_0, label_1, image_paths_1)     
            
            labels = np.concatenate((label_0,label_1))
            features = np.concatenate((features_0,features_1))
            # features = np.array(features)
            print('features.shape===============',features.shape)
            self.save_hog_features_and_labels(features, label, obj_id, feature_dir)
            labels = []
            features = []
                
class AccuracyHistory(Callback):
    def __init__(self):
        super().__init__()
        self.reset_dict()
        self.start_time = time.time()
             
    def reset_dict(self):
        self.no_of_samples = []
        self.model=[]
        self.loss = []
        self.max_iterations = []
        self.tpr = []
        self.tnr = []
        self.f1 = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.elapsed_time = []
        self.validation_data = None  
       
    def set_model(self, model):
        self.model = model
        if hasattr(self.model, 'validation_data') and self.model.validation_data:
            self.validation_data = (self.model.validation_data[0], self.model.validation_data[1])
        else:
            print("Validation data is not available at the start of training.")    
     
    def on_epoch_end(self, epoch, logs=None):
        
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        self.no_of_samples.append(tune.no_of_samples)
        self.model.append(tune.model)
        self.loss.append(tune.loss)
        self.max_iterations.append(tune.max_iterations)
        self.tpr.append(tune.tpr)
        self.tnr.append(tune.tnr)
        self.f1.append(tune.f1)
        self.accuracy.append(tune.accuracy)
        self.precision.append(tune.precision)
        self.recall.append(tune.recall)
        self.elapsed_time.append(elapsed_time)

        
    def create_accuracy_dataframe(self):
        accuracy_df = pd.DataFrame({
            
            'no_of_samples':self.no_of_samples,
            'model':self.model,
            'loss':self.loss,
            'max_iterations':self.max_iterations,
            'tpr':self.tpr,
            'tnr':self.tnr,
            'f1':self.f1,
            'precision':self.precision,
            'recall':self.recall,
            'accuracy':self.accuracy,
            'elapsed_time':self.elapsed_time
        })
        return accuracy_df    
    def save_to_csv(self, accuracy_df):
            # Start with summary1.csv
            file_number = 0
            while True:
                filename = 'tune_log_ml/'+'summary' + str(file_number) + '.csv'
                # Check if the file already exists
                if not os.path.isfile(filename):
                    break
                file_number += 1
            filename_model ='tune_log_ml/'+ 'model' + str(file_number) + '.joblib'
            joblib.dump(network.model, filename_model)
            accuracy_df.to_csv(filename, index=False)    
        
class tuning():
    def __init__(self):

        self.no_of_samples = 0
        self.model='SGDClassifier'
        self.loss = 'log_loss'
        self.max_iterations = 1

        
        
    
    def define_dataset(self,no_of_train_samples=1000000, no_of_test_samples=1000000):

        # Example usage
        train_features_dir = '/home/rag-tt/workspace/hog_features4/train_features'
        test_features_dir = '/home/rag-tt/workspace/hog_features3/test_features'

        batch_size = 128

        train_data_loader = create_data_loader(train_features_dir, batch_size)
        test_data_loader = create_data_loader(test_features_dir, batch_size)
        return train_data_loader, test_data_loader 
                        
    def start_training(self):
        try:
            train_data_loader, test_data_loader  = self.define_dataset()
            
            #print the tuning parametrs before training
            # accuracy_history.on_epoch_end(0)
            # df = accuracy_history.create_accuracy_dataframe()
            # # Transpose the DataFrame
            # df_transposed = df.transpose()
            # print(df_transposed)

            network.train(train_data_loader)
            self.tpr, self.tnr, self.f1, self.accuracy, self.precision, self.recall=  network.evaluate(test_data_loader)
            
            # Print the values in a formatted statement
            print(f"Metrics Summary:")
            print(f"Accuracy: {self.accuracy:.4f}")
            print(f"Precision: {self.precision:.4f}")
            print(f"Recall: {self.recall:.4f}")
            print(f"F1-Score: {self.f1:.4f}")
            print(f"True Positive Rate (TPR): {self.tpr:.4f}")
            print(f"True Negative Rate (TNR): {self.tnr:.4f}")
            
            accuracy_history.on_epoch_end(0)
            accuracy_df = accuracy_history.create_accuracy_dataframe()
            accuracy_history.save_to_csv(accuracy_df)
        
        # Ensure accuracy data is saved even if training is interrupted 
        finally:        
            # # Create a DataFrame from the accuracy history lists
            # accuracy_df = accuracy_history.create_accuracy_dataframe()

            # # Save the DataFrame to a CSV file
            # accuracy_history.save_to_csv(accuracy_df)
            # accuracy_history.reset_dict()             
            print('done')
    def Tune(self):
         self.start_training()

        
def true_positives(y_true, y_pred):
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    return tp

def true_negatives(y_true, y_pred):
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    return tn

def false_positives(y_true, y_pred):
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    return fp

def false_negatives(y_true, y_pred):
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)
    return fn

def list_subdirectories(directory):
    try:
        # Get the list of all entries in the directory
        entries = os.listdir(directory)
        
        # Filter out and list only the directories
        subdirs = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        
        return subdirs
    except FileNotFoundError:
        return f"The directory '{directory}' does not exist."
    except PermissionError:
        return f"Permission denied to access '{directory}'."
   
   
def load_hog_features_and_labels2(features_dir):
    # Convert to string if it's in bytes
    features_dir = features_dir.decode('utf-8') if isinstance(features_dir, bytes) else str
    
    features_files = sorted(glob(os.path.join(features_dir, 'features_*.npy')))
    # print('*********************************** file=',features_files)
    labels_files = sorted(glob(os.path.join(str(features_dir), 'labels_*.npy')))
    
    # Initialize lists to store all features and labels across files (NEW)
    all_features = []
    all_labels = []
    for features_file, labels_file in zip(features_files, labels_files):
        features = np.load(features_file, allow_pickle=True)
        labels = np.load(labels_file)
        # Append each batch of features and labels to the lists (NEW)
        all_features.append(features)
        all_labels.append(labels)
        
        # for feature, label in zip(features, labels):
        #     yield feature, label
        
    # Concatenate all batches into a single array for both features and labels (NEW)
    all_features = np.concatenate(all_features, axis=0)  # Combine along batch dimension
    all_labels = np.concatenate(all_labels, axis=0)

    # Yield each feature and label pair after concatenation (NEW)
    for feature, label in zip(all_features, all_labels):
        yield feature, label
        

#memory efficient load hog features
def load_hog_features_and_labels(features_dir):
    # Convert to string if it's in bytes
    features_dir = features_dir.decode('utf-8') if isinstance(features_dir, bytes) else str
    
    features_files = sorted(glob(os.path.join(features_dir, 'features_*.npy')))
    np_features_files = np.array(features_files)
    # print('featire==================', features_files)
    labels_files = sorted(glob(os.path.join(str(features_dir), 'labels_*.npy')))
    
    for features_file, labels_file in zip(features_files, labels_files):
        # Load one file at a time with memory mapping to avoid loading everything into memory at once
        features = np.load(features_file, mmap_mode='r')  # Memory map to avoid high memory usage
        labels = np.load(labels_file)  # Labels are usually smaller, no need to memory map here
        # print('features================', features)
        # # Ensure that each individual sample (not the full batch) is yielded
        # for i in range(features.shape[0]):  # Iterate over the batch dimension
        #     feature = features[i]  # Get each individual feature of shape (8, 1944)
        #     label = labels[i]  # Corresponding label
        #     # print('feature_shape====================================================',features.shape)
        
        for feature, label in zip(features, labels):    
        # Yield each feature and label pair as you process them
        # for feature, label in zip(features, labels):
            if feature.shape != (8, 1946):
                raise ValueError(f"Expected feature shape (8, 1946), but got {feature.shape[-2:]}")
            yield feature, label  # Yielding as you go instead of concatenating all
            
        feature_np =  np.array(feature)
        # print('feature=****************************************************',feature_np.shape)
            
class HOGFeaturesDataset(tf.data.Dataset):
    def _generator(features_dir):
        yield from load_hog_features_and_labels(features_dir)

    def __new__(cls, features_dir):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(8,1946), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64)
            ),
            args=(features_dir,)
        )

def create_data_loader(features_dir, batch_size, shuffle=False):
    dataset = HOGFeaturesDataset(features_dir)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
class create_network():

    def __init__(self):
        self.x =0
        self.model = SGDClassifier(loss='log_loss',  # Use logistic regression for binary classification
                                   max_iter=1,  # One iteration per batch
                                   warm_start=True)  # Keep weights between batches (important) 
        self.latest_valid_batch = 0
        
    def train(self, data_loader):
        # Iterate over the data loader

        # Initialize deque to keep track of the last 30 batch accuracies
        accuracy_deque = deque(maxlen=30)
        

        completed_batches = 0
        # Create an imputer instance to fill NaN values with the mean (or another strategy)
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')        
        # Iterate over the data loader
        for batch_features, batch_labels in data_loader:  
              
            # Flatten the features to fit into the model
            batch_features_flat = tf.reshape(batch_features, (batch_features.shape[0], -1)).numpy()  # Flatten to 2D
            batch_labels_flat = batch_labels.numpy()
            # Impute missing values in batch_features_flat
            batch_features_flat = imputer.fit_transform(batch_features_flat)
            if batch_features_flat.shape == (128,15568):
                self.latest_valid_batch = batch_features_flat
            else:
                batch_features_flat = self.latest_valid_batch
                
            if batch_labels_flat.shape == (128,):
                self.latest_valid_labels = batch_labels_flat
            else:
                batch_labels_flat = self.latest_valid_labels
                                
            print('batch_features=', batch_features_flat.shape)
            print('batch_lables=', batch_labels_flat.shape)
            # Fit the model incrementally with the current batch
            self.model.partial_fit(batch_features_flat, batch_labels_flat,classes=np.array([0, 1]))

            # Calculate accuracy on the current batch
            current_predictions = self.model.predict(batch_features_flat)
            accuracy = accuracy_score(batch_labels_flat, current_predictions)

            # Append current accuracy to the deque
            accuracy_deque.append(accuracy)

            # Calculate the average accuracy over the last 30 batches
            average_accuracy = np.mean(accuracy_deque)
            
             # Increment the completed batches counter
            completed_batches += 1

            # Print current status
            print(f"Completed Batch {completed_batches}: Current Batch Accuracy: {accuracy:.4f}, Average Accuracy (last 30 batches): {average_accuracy:.4f}")
        print("Model training completed.")
    def evaluate(self, data_loader):
        # Initialize accumulators for performance metrics
        total_samples = 0
        correct_predictions = 0
        tn, fp, fn, tp = 0, 0, 0, 0
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  
        
        for batch_features, batch_labels in data_loader:
            # Flatten features for the model
            batch_features_flat = tf.reshape(batch_features, (batch_features.shape[0], -1)).numpy()
                        # Impute missing values in batch_features_flat
            batch_features_flat = imputer.fit_transform(batch_features_flat)
            batch_labels_flat = batch_labels.numpy() 
            if batch_features_flat.shape == (128,15568):
                self.latest_valid_batch = batch_features_flat
            else:
                batch_features_flat = self.latest_valid_batch
                
            if batch_labels_flat.shape == (128,):
                self.latest_valid_labels = batch_labels_flat
            else:
                batch_labels_flat = self.latest_valid_labels
                
            # Predict for the batch
            batch_predictions = self.model.predict(batch_features_flat)
            batch_predictions = (batch_predictions > 0.5).astype(int)  # Assuming binary classification

            # Update total samples and correct predictions for accuracy calculation
            correct_predictions += np.sum(batch_predictions == batch_labels.numpy())
            total_samples += batch_labels.shape[0]

            # Update confusion matrix values
            batch_tn, batch_fp, batch_fn, batch_tp = confusion_matrix(batch_labels.numpy(), batch_predictions, labels=[0, 1]).ravel()
            tn += batch_tn
            fp += batch_fp
            fn += batch_fn
            tp += batch_tp

        # Calculate final accuracy
        accuracy = correct_predictions / total_samples

        # Calculate TPR, TNR, Precision, Recall, F1
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr  # Recall is the same as TPR in binary classification
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Output performance metrics
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"TPR (Recall/Sensitivity): {tpr:.4f}")
        print(f"TNR (Specificity): {tnr:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return tpr, tnr, f1, accuracy, precision, recall


network = create_network()
tune = tuning()
accuracy_history = AccuracyHistory()
tune.Tune()