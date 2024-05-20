import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential 
import pandas as pd
import numpy as np
import pathlib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

class Manage_data():
    def __init__(self):
        data_dir='/app/tactile_images/'
        self.data_dir= pathlib.Path(data_dir)

    def find_no_of_images(self, obj_id):
        image_dir = os.path.join(self.data_dir, str(obj_id))
        image_dir= pathlib.Path(image_dir)
        no_of_images= len(list(image_dir.glob('*.jpg')))
        return no_of_images
    
    def parse_function(self, filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [480, 640])  # Adjust size as needed
        return image_resized, label

    
    def load_data(self, data_dir, no_of_samples = 50):
        file_paths = []
        image_paths = []
        y = []
        window_size = 5
        for obj_id in range(no_of_samples):
            no_of_images = self.find_no_of_images(obj_id)
            if no_of_images < 40:
                continue
            
            csv_path = os.path.join(data_dir, str(obj_id),'slip_log.csv')
            label = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=1, dtype=None, encoding=None)
            y.append(label)
            for img_id in range(no_of_images):
                image_path = os.path.join(data_dir, str(obj_id), str(img_id)+ '.jpg')
                file_paths.append(image_path)

        y = np.concatenate(y)
        y = np.array(y)
        file_paths = np.array(file_paths)
        return file_paths, y

    def parse_function_sequential(self, filenames, label):
        images = []
        for filename in filenames:
            image_string = tf.io.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image_resized = tf.image.resize(image_decoded, [480, 640])  # Adjust size as needed
            # Ensure images are float32 and normalized between 0 and 1
            images.append(image_resized)
        images = tf.stack(images)
        return images, label
    
    def load_sequential_data(self, no_of_samples = 50):
        file_paths = []
        image_paths = []
        sequential_image_paths = []
        y = []
        window_size = 5
        for obj_id in range(no_of_samples):
            no_of_images = self.find_no_of_images(obj_id)
            if no_of_images < 40:
                continue
            
            csv_path = os.path.join(self.data_dir, str(obj_id),'slip_log.csv')
            label = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=1, dtype=None, encoding=None)
            y.append(label[:-4])
            
            for img_id in range(no_of_images):
                image_path = os.path.join(self.data_dir, str(obj_id), str(img_id)+ '.jpg')
                image_paths.append(image_path)
            for i in range(0, len(image_paths) - 4):  # Ensuring sequences of 5 images
                row = image_paths[i:i+5]
                sequential_image_paths.append(row)
            image_paths = []

        y = np.concatenate(y)
        y = np.array(y)

        file_paths = np.array(sequential_image_paths)

        return file_paths, y
    
    def create_sequential_dataset(self,file_paths, labels):
                # Create a TensorFlow dataset from the file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        
        # # Map the function to each sequence of file paths and labels
        # dataset = dataset.map(
        #     lambda file_paths, label: self.parse_function_sequential(file_paths, label),
        #     num_parallel_calls=tf.data.AUTOTUNE
        # )
        def wrapped_parse_function(filenames, label):
            images, label = tf.py_function(func=self.parse_function_sequential, inp=[filenames, label], Tout=[tf.float32, tf.int64])
            images.set_shape((5, 480, 640, 3))  # Explicitly set the shape
            label.set_shape([])  # Explicitly set the shape for the label
            return images, label
        # # Map the parse_function to the dataset using tf.py_function
        # dataset = dataset.map(lambda file_paths, label: tf.py_function(func=self.parse_function_sequential, inp=[file_paths, label], Tout=[tf.float32, tf.int64]),
        #               num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.map(wrapped_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        #     # Use tf.py_function to apply the parse_function
        # def wrapped_parse_function(filenames, label):
        #     return tf.py_function(func=self.parse_function_sequential, inp=[filenames, label], Tout=[tf.float32, tf.float32])

        # dataset = dataset.map(wrapped_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(32)  # Adjust batch size as needed
        self.dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def create_dataset(self):
        file_paths, labels = self.load_data(self.data_dir)
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        batch_size = 32  # Adjust batch size as needed
        dataset = dataset.batch(batch_size)
        self.dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def split_dataset(self, file_paths):
        dataset_size = len(file_paths)
        train_size = int(0.8 * dataset_size)
        self.train_dataset = self.dataset.take(train_size)
        self.val_dataset = self.dataset.skip(train_size)

class create_network():

    def __init__(self):
        self.x =0
    def cnn_lstm1(self):

        # Define CNN model
        cnn_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten()  # Flatten the spatial dimensions
        ])

        
        # Define LSTM model
        lstm_model = Sequential([
            LSTM(64,input_shape=(5, 144768) ),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])

        # Combine CNN and LSTM models
        self.model = Sequential([
            TimeDistributed(cnn_model, input_shape=(5, 480, 640, 3)),  # Apply CNN to each frame in the sequence
            (Reshape((5,144768))),
            lstm_model,
        ])
        self.model.summary()

    def train(self, train_dataset, val_dataset):
        cp = ModelCheckpoint('model/', save_best_only=True)
        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])
        self.model.fit(train_dataset,validation_data=val_dataset, epochs=30, callbacks=[cp])

manage_data = Manage_data()
network = create_network()
if __name__ == "__main__":
        # Set TensorFlow logging level to suppress detailed messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR

    # Initialize TensorFlow after setting environment variables
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


        # creats a numpy array of images, then it lumps the images to together in batch of 5 for lstm
    #(None,5)
    filepaths, label = manage_data.load_sequential_data()
 

    # creates a tensor called dataset which contains the images. 
    #The images are not loaded and stored in the seperate memory, it uses the existing images instead
    #This saves time and memory
    #(None,5,480,640,3)
    print(filepaths.shape)
    manage_data.create_sequential_dataset(filepaths,label)
    for batch in manage_data.dataset.take(1):  # Take one batch to print its shape
        images_batch, labels_batch = batch
        print("Images batch shape:", images_batch.shape)
        print("Labels batch shape:", labels_batch.shape)
    manage_data.split_dataset(filepaths)
    #creates a combined network of cnn and lstm
    print('train set=',manage_data.train_dataset)
    network.cnn_lstm1()
    network.train(manage_data.train_dataset, manage_data.val_dataset)
    
    