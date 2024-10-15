import os
# Set the environment variable to use only the GPU with ID 1 (GTX 1080 Ti)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from memory_profiler import profile
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
logdir ='logdir'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
tf.profiler.experimental.start('logdir')
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
        
        # Use tf.stack only if images list is not empty
        if images:
            images = tf.stack(images)
        else:
            images = tf.zeros((tune.img_sequence_window_size, 224, 224, 3))  # Handle empty case
        
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

        # Add batching
        dataset = dataset.batch(tune.batch_size)  # Adjust batch size as needed
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
        
class create_network():

    def __init__(self):
        self.x =0
        
    def cnn_lstm1(self):

        # Define CNN model
        cnn_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 3),kernel_regularizer=l1(tune.regularizaion_const)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten()  # Flatten the spatial dimensions
        ])

        
        # Define LSTM model
        lstm_model = Sequential([
            LSTM(64,input_shape=(tune.img_sequence_window_size, 144768),kernel_regularizer=l1(tune.regularizaion_const) ),
            Dense(8, activation='relu', kernel_regularizer=l1(tune.regularizaion_const)),
            Dense(1, activation='sigmoid'),
        ])

        # Combine CNN and LSTM models
        self.model = Sequential([
            TimeDistributed(cnn_model, input_shape=(tune.img_sequence_window_size, 480, 640, 3)),  # Apply CNN to each frame in the sequence
            (Reshape((tune.img_sequence_window_size,144768))),
            lstm_model,
        ])
        self.model.summary()

    def vgg_lstm(self):
        # VGG16 model with pre-trained weights
        #include top  false remove the final classification layer
        vgg_model = tf.keras.applications.VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
        # Freeze the VGG16 layers if you don't want to train them
        
        # Retain only the first n layers of the VGG16 model
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[tune.vgg_layers-1].output)
      
        for layer in vgg_model.layers:
            layer.trainable = False
            
        additional_conv = Sequential([
            vgg_model,
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2))
        ])

        # Define CNN model
        vgg_model_flatten = Sequential([
            additional_conv,
            # Flatten(),  # Flatten the spatial dimensions
            GlobalAveragePooling2D(),  # Use Global Average Pooling instead of Flatten
            # Dense(tune.dense_neurons1, activation='relu'),
            # Dropout(tune.dropout1)
        ])
        
        vgg_model_flatten.summary()
        #25088 is the output of vff_model_flatten
        # Define LSTM model
        lstm_model = Sequential([
            LSTM(64, input_shape=(tune.img_sequence_window_size, tune.dense_neurons1)),
            Dropout(tune.dropout2),  # Dropout layer to prevent overfitting
            Dense(tune.dense_neurons2, activation='relu'),
            Dropout(tune.dropout3),
            Dense(1, activation='sigmoid'),
                ])

        # Combine CNN and LSTM models
        self.model = Sequential([
            TimeDistributed(vgg_model_flatten, input_shape=(tune.img_sequence_window_size, 224, 224, 3)),  # Apply CNN to each frame in the sequence
            (Reshape((tune.img_sequence_window_size,tune.dense_neurons1))),
            lstm_model,
        ])
        vgg_model.summary()
        
        self.model.summary()
   
    def train(self, train_dataset, val_dataset):
        cp = ModelCheckpoint(filepath='checkpoint.h5',monitor='val_accuracy',save_best_only=True,save_weights_only=False,verbose=1)        # Save the full model (structure + weights).verbose=1)
            # EarlyStopping callback to stop training when validation accuracy stops improving
        es = EarlyStopping(monitor='val_accuracy', patience=1, restore_best_weights=True)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensor board
        tb= tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                # Shuffle training dataset before each epoch
        # train_dataset_shuffled = train_dataset.shuffle(buffer_size=train_dataset.cardinality(), reshuffle_each_iteration=tune.reshuffle)
        self.model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=tune.learning_rate),metrics=['accuracy', true_positives, false_positives, true_negatives, false_negatives])
        self.model.fit(train_dataset,validation_data=val_dataset, epochs=tune.epochs, callbacks=[cp,es,accuracy_history], verbose =1)
        
class AccuracyHistory(Callback):
    def __init__(self):
        super().__init__()
        self.reset_dict()
        self.start_time = time.time()
                     
    def reset_dict(self):
        self.epoch_count = []
        self.train_accuracy = []
        self.val_accuracy = []
        self.img_sequence_window_size = []
        self.stride = []
        self.learning_rate = []
        self.reshuffle =  []
        self.dropout1 = []
        self.dropout2 = []
        self.dropout3 = []
        self.dropout4 = []
        self.regularization_constant = []
        self.batch_size = []
        self.dense_neurons1 =[]
        self.dense_neurons2 =[]
        self.no_of_samples = []
        self.epochs  = []
        self.vgg_layers = []
        self.other_param = []
        self.no_of_nonslip_data = []
        self.slip_instant_labels = [] 
        self.max_labels = []
        self.min_trim_value = []
        self.tp = []
        self.tn = []
        self.fp = []
        self.fn = []
        self.tpr = []
        self.tnr = []
        self.fnr = []
        self.f1 = []
        self.elapsed_time = []
        self.validation_data = None  
       
    def set_model(self, model):
        self.model = model
        if hasattr(self.model, 'validation_data') and self.model.validation_data:
            self.validation_data = (self.model.validation_data[0], self.model.validation_data[1])
        else:
            print("Validation data is not available at the start of training.")    
     
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {'accuracy': 0, 'val_accuracy': 0,'true_positives':0,'true_negatives':0,'false_positives':0,'false_negatives':0 }
        

        tp = logs.get('true_positives')
        tn = logs.get('true_negatives')
        fp = logs.get('false_positives')
        fn = logs.get('false_negatives')


        # Compute TPR, TNR, and F1 score
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        f1 =  (tp + tn) /(tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0
        # print('tp= ',tp,'tn= ',tn,'fp= ',fp,'fn= ',fn)
        # print('tpr= ',tpr,'fnr= ',fnr,'f1= ',f1)
        
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        
        self.epoch_count.append(epoch + 1)
        self.train_accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.img_sequence_window_size.append(tune.img_sequence_window_size)
        self.stride.append(tune.stride)
        self.learning_rate.append(tune.learning_rate)
        self.reshuffle.append(tune.reshuffle)
        self.dropout1.append(tune.dropout1)
        self.dropout2.append(tune.dropout2)
        self.dropout3.append(tune.dropout3)
        self.dropout4.append(tune.dropout4)
        self.regularization_constant.append(tune.regularization_constant)
        self.batch_size.append(tune.batch_size)
        self.dense_neurons1.append(tune.dense_neurons1)
        self.dense_neurons2.append(tune.dense_neurons2)
        self.no_of_samples.append(tune.no_of_samples)
        self.epochs.append(tune.epochs )
        self.vgg_layers.append(tune.vgg_layers)
        self.other_param.append(tune.other_param)
        self.no_of_nonslip_data.append(tune.no_of_nonslip_data)
        self.slip_instant_labels.append(tune.slip_instant_labels)
        self.max_labels.append(tune.max_labels)
        self.min_trim_value.append(tune.min_trim_value)
        self.tp.append(tp)
        self.tn.append(tn)
        self.fp.append(fp)
        self.fn.append(fn)
        self.tpr.append(tpr)
        self.fnr.append(fnr)
        self.f1.append(f1)
        self.elapsed_time.append(elapsed_time)
        
    def create_accuracy_dataframe(self):
        accuracy_df = pd.DataFrame({
            'Epoch': self.epoch_count,
            'Train_Accuracy': self.train_accuracy,
            'Val_Accuracy': self.val_accuracy,
            'img_sequence_window_size': self.img_sequence_window_size,
            'stride':self.stride,
            'Learning_Rate': self.learning_rate,
            'Reshuffle': self.reshuffle,
            'Dropout1': self.dropout1,
            'Dropout2': self.dropout2,
            'Dropout3': self.dropout3,
            'Dropout4': self.dropout4,
            'Regularization_Constant': self.regularization_constant,
            'Batch_Size': self.batch_size,
            'dense_neurons1': self.dense_neurons1,
            'dense_neurons2': self.dense_neurons2,
            'no_of_samples':self.no_of_samples,
            'epochs':self.epochs, 
            'vgg_layers':self.vgg_layers,
            'other_param':self.other_param,
            'no_of_nonslip_data':self.no_of_nonslip_data,
            'slip_instant_labels':self.slip_instant_labels,
            'max_labels':self.max_labels,
            'min_trim_value':self.min_trim_value,
            'tp':self.tp,
            'tn':self.tn,
            'fp':self.fp,
            'fn':self.fn,
            'tpr':self.tpr,
            'fnr':self.fnr,
            'f1':self.f1,
            'elapsed_time':self.elapsed_time
        })
        return accuracy_df    
    def save_to_csv(self, accuracy_df):
            tf.profiler.experimental.stop()
            # Start with summary1.csv
            file_number = 0
            while True:
                filename = 'tune_log/'+'summary' + str(file_number) + '.csv'
                # Check if the file already exists
                if not os.path.isfile(filename):
                    break
                file_number += 1
            filename_model ='tune_log/'+ 'model' + str(file_number) + '.h5'
            network.model.save(filename_model)
            accuracy_df.to_csv(filename, index=False)    
        
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
        self.batch_size = 2
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
            
            train_labels, train_file_paths = manage_data.load_data(data_dir = train_data_dir, no_of_samples=1000)
            test_labels, test_file_paths = manage_data.load_data(data_dir = test_data_dir, no_of_samples=1000)
            
            train_labels, train_file_paths = manage_data.shuffle_file_paths(train_labels, train_file_paths)
            train_labels = np.array(train_labels)
            test_labels = np.array(test_labels)
            print('train_labels_type',train_labels.dtype)
            print('test labels dtype',test_labels.dtype )
            
            train_dataset = manage_data.create_dataset(train_labels, train_file_paths )
            test_dataset = manage_data.create_dataset(test_labels, test_file_paths) 
            return train_dataset, test_dataset 
                        
    def start_training(self):
        try:
            train_dataset, test_dataset = self.define_dataset()
            network.vgg_lstm()
            
            #print the tuning parametrs before training
            accuracy_history.on_epoch_end(0)
            df = accuracy_history.create_accuracy_dataframe()
            # Transpose the DataFrame
            df_transposed = df.transpose()
            print(df_transposed)
            network.train(train_dataset, test_dataset)
        
        # Ensure accuracy data is saved even if training is interrupted 
        finally:        
            # Create a DataFrame from the accuracy history lists
            accuracy_df = accuracy_history.create_accuracy_dataframe()

            # Save the DataFrame to a CSV file
            accuracy_history.save_to_csv(accuracy_df)
            accuracy_history.reset_dict()             
                   
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
network = create_network()
manage_data= Manage_data()


tune = tuning()
accuracy_history = AccuracyHistory()

tune.Tune()
# test_function()