import os
# Set the environment variable to use only the GPU with ID 1 (GTX 1080 Ti)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.keras.models import load_model
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
import pandas as pd
import numpy as np
import pathlib
import time

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

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     except RuntimeError as e:
#         print(e)
        
import logging
# Set the print options to display the entire array
np.set_printoptions(threshold=np.inf)

class Manage_data():
    def __init__(self):
        data_dir='/home/rag-tt/workspace'
        self.train_data_dir = os.path.join(data_dir,'train_data')
        self.test_data_dir = os.path.join(data_dir,'test_data')
        self.data_dir= pathlib.Path(data_dir)
        self.last_valid_image = None  # Initialize to store the last valid image
        self.create_a_dummy_image()
        
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
                
    def load_data(self,data_dir, no_of_samples):
        file_paths = []
        image_paths = []
        y = []
        
        for obj_id in range(no_of_samples):
            no_of_images = self.find_no_of_images(data_dir,obj_id)
            csv_path = os.path.join(data_dir, str(obj_id),'slip_log.csv')
            if no_of_images < 40 or not os.path.exists(csv_path):
                continue
            label = self.create_slip_instant_labels(csv_path)
            
            label2 = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=1, dtype=None, encoding=None)
            
            for img_id in range(no_of_images):
                image_path = os.path.join(data_dir, str(obj_id), str(img_id)+ '.jpg')
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

            
            # club images together seperately and then jouin together in one dataset
            clubbed_image_paths = []
            for i in range(0, len(image_paths_0) - (tune.img_sequence_window_size-1), tune.stride):  # Ensuring sequences of 5 images
                row = image_paths_0[i:i+tune.img_sequence_window_size]
                clubbed_image_paths.append(row)
                
            for i in range(0, len(image_paths_1) - (tune.img_sequence_window_size-1), tune.stride):  # Ensuring sequences of 5 images
                row = image_paths_1[i:i+tune.img_sequence_window_size]
                clubbed_image_paths.append(row)
                            
            image_paths = []
            label_0 = np.array(label_0[(tune.img_sequence_window_size-1):])
            label_0 =  label_0[::tune.stride]
            label_1 = np.array(label_1[(tune.img_sequence_window_size-1):])
            label_1 =  label_1[::tune.stride]
            label = np.concatenate((label_0, label_1))
            
            '''   
            new version - label = label[(tune.img_sequence_window_size-1):]       
            example values of label and clubbed_image_paths  
            here img_sequence_window_size = 3   
            clubbed_image_paths = [0,1,2] ... [80,81,82]       
            label size = 81, imagepaths_size = 81
            zeroes = 39, ones = 42
            [38,39,40] == 0 ,[39,40,41] == 1, [40,41,42] == 1, [41,42,43] == 1
            '''
            
            '''   
            striding-
            1 = [0,1,2],[1,2,3],..., obj_id=2, image_paths_size=135, label_size = 135
            2 = [0,1,2],[2,3,4],..., obj_id=2, image_paths_size=68 135/2 = 67.5, label_size = 68
            3 = [0,1,2],[3,4,5],..., obj_id=2, imagepaths_size=45,  135/3 = 45, label_size = 45
            '''
            # duplicate the data to balance the ones and zeroes 
            # label, clubbed_image_paths = self.duplicate_n_balance_data(label, clubbed_image_paths)
            # for i in range(label.size):
            #     print('data=', clubbed_image_paths[i])
            #     print('label=', label[i])
            
            
            clubbed_image_paths_np = np.array(clubbed_image_paths)
            shape = clubbed_image_paths_np.shape
            shape_np = np.array(shape)
            
            # remove arrays with inconsistent shapes
            if shape_np.shape[0] != 2:
                continue

                
            y.append(label)
            file_paths.append(clubbed_image_paths)
        #concatenate = merge multipe arrays into one
        y = np.concatenate(y)
        labels = np.array(y)
        
        # print(self.labels.shape) = 2025
        file_paths = np.concatenate(file_paths)
        # print(self.file_paths.shape) = (2025,3)
        image_paths = []
        del y
        return labels, file_paths
        
    def shuffle_file_paths(self, labels, file_paths):
        # Shuffle the dataset
        indices = np.arange(len(file_paths))
        np.random.shuffle(indices)
        file_paths = file_paths[indices]
        labels = labels[indices]
        return labels, file_paths

    def create_a_dummy_image(self):
        filename= os.path.join(self.train_data_dir,str(1), str(1) + '.jpg')
        image_string = tf.io.read_file(filename)                                        
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [224, 224])  # Resize as needed
        
        # Convert image to float32 and preprocess for VGG16
        image = tf.cast(image_resized, tf.float32)
        image = preprocess_input(image)  # Normalize for VGG16
        
        self.last_valid_image = image  # Update last valid image
            
    def parse_function_vgg(self, filenames, label):
        images = []
        
        for filename in filenames:
            image_string = tf.io.read_file(filename)
            
            # Check if the image is empty
            if tf.equal(tf.size(image_string), 0):
                print(f"Warning: File {filename} is empty or invalid.")
                if self.last_valid_image is not None:
                    print(f"Using last valid image as placeholder for {filename}.")
                    images.append(self.last_valid_image)
                continue
            
            # Decode JPEG and handle potential decoding errors
            try:
                image_decoded = tf.image.decode_jpeg(image_string, channels=3)
                image_resized = tf.image.resize(image_decoded, [224, 224])  # Resize as needed
                
                # Convert image to float32 and preprocess for VGG16
                image = tf.cast(image_resized, tf.float32)
                image = preprocess_input(image)  # Normalize for VGG16
                
                images.append(image)
            
            except tf.errors.InvalidArgumentError:
                print(f"Error: Failed to decode JPEG file at {filename}.")
                if self.last_valid_image is not None:
                    print(f"Using last valid image as placeholder for {filename}.")
                    images.append(self.last_valid_image)  # Use last valid image
                continue
            except Exception as e:
                print(f"Unexpected error with file {filename}: {e}")
                if self.last_valid_image is not None:
                    print(f"Using last valid image as placeholder for {filename}.")
                    images.append(self.last_valid_image)  # Use last valid image
                continue
        images = tf.stack(images)
        return images, label
        
    def create_dataset(self, labels, file_paths):
                # Create a TensorFlow dataset from the file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        

        def wrapped_parse_function(filenames, label):
            images, label = tf.py_function(func=self.parse_function_vgg, inp=[filenames, label], Tout=[tf.float32, tf.int64])
            images.set_shape((tune.img_sequence_window_size, 224, 224, 3))  # Explicitly set the shape
            label.set_shape([])  # Explicitly set the shape for the label
            return images, label
 
        
        dataset = dataset.map(wrapped_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(tune.batch_size)  # Adjust batch size as needed
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
        
        
        
def test(val_dataset):
    custom_objects = {
    'true_positives': true_positives,
    'true_negatives': true_negatives,
    'false_positives': false_positives,
    'false_negatives': false_negatives
    }
    model = load_model('checkpoint2.h5', custom_objects)
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=1)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_accuracy}')
    # Create a string with the results
    result_text = f'Validation Loss: {val_loss}\nValidation Accuracy: {val_accuracy}\n'

    # Save the results to a text file
    with open('validation_results.txt', 'w') as file:
        file.write(result_text)
        
class tuning():
    def __init__(self):
        self.img_sequence_window_size_array = [8, 10, 12]
        self.learning_rate_array = [0.00003, 0.00005]
        self.reshuffle_array=[False, True]
        self.regularization_constant_array = [0.01, 0.05, 0.1, 0.2, 0.3]
        self.dense_neurons2_array = [8, 16, 32]
        self.vgg_layers_array= [7,11,15,19]
        self.slip_instant_labels_array = [0.0001,0.0005, 0.001, 0.003, 0.005]
        
        self.img_sequence_window_size =  self.img_sequence_window_size_array[0]
        self.stride = 1 
        self.learning_rate = self.learning_rate_array[0]
        self.reshuffle =  self.reshuffle_array[0]
        self.dropout1 = 0.5
        self.dropout2 = 0.5
        self.dropout3 = 0.5
        self.dropout4 = 0.5
        self.regularization_constant = 0.001
        self.batch_size = 8
        self.dense_neurons1 = 64
        self.dense_neurons2 = 8
        self.csv_id = 0
        self.no_of_samples = 50
        self.epochs = 40
        self.vgg_layers = 19
        self.other_param='additional cnn + global average'
        self.no_of_nonslip_data = 2000
        self.slip_instant_labels = 0.0003
        self.max_labels = 0.02
        self.min_trim_value = 0.000007
    
    def define_dataset(self,no_of_train_samples=1000000, no_of_test_samples=1000000):

            train_data_dir = manage_data.train_data_dir
            test_data_dir = manage_data.test_data_dir
            
            train_data_qty = manage_data.count_subdirectories(train_data_dir)
            test_data_qty = manage_data.count_subdirectories(test_data_dir)
            if no_of_train_samples > train_data_qty:
                no_of_train_samples = train_data_qty
                self.no_of_samples = train_data_qty
            if no_of_test_samples > test_data_qty:
                no_of_test_samples = test_data_qty
            
            train_labels, train_file_paths = manage_data.load_data(data_dir = train_data_dir, no_of_samples=15)
            test_labels, test_file_paths = manage_data.load_data(data_dir = test_data_dir, no_of_samples=test_data_qty)
            
            train_labels, train_file_paths = manage_data.shuffle_file_paths(train_labels, train_file_paths)
            train_labels = np.array(train_labels)
            test_labels = np.array(test_labels)
            print('train_labels_type',train_labels.dtype)
            print('test labels dtype',test_labels.dtype )
            
            train_dataset = manage_data.create_dataset(train_labels, train_file_paths )
            test_dataset = manage_data.create_dataset(test_labels, test_file_paths) 
            return train_dataset, test_dataset 
                        
    def start_training(self):
         train_dataset, test_dataset = self.define_dataset()
         test(test_dataset)
        

    def Tune(self):
        # for value in self.vgg_layers_array:
        #     self.vgg_layers = value         
        #     self.start_training()
        # self.vgg_layers= 19
        
        for value in self.learning_rate_array:
            self.learning_rate = value           
            self.start_training()
        self.learning_rate = 0.00003
        

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
        return "The directory"+ str(directory)+ " does not exist."
    except PermissionError:
        return "Permission denied to access" + str(directory)
    
def test_function():
    train_dataset, test_dataset = tune.define_dataset(2,1)
    
    
manage_data = Manage_data()
# network = create_network()
tune = tuning()
tune.Tune()
# test_function()