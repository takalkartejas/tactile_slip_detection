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
    labels_files = sorted(glob(os.path.join(str(features_dir), 'labels_*.npy')))
    
    for features_file, labels_file in zip(features_files, labels_files):
        # Load one file at a time with memory mapping to avoid loading everything into memory at once
        features = np.load(features_file, mmap_mode='r')  # Memory map to avoid high memory usage
        labels = np.load(labels_file)  # Labels are usually smaller, no need to memory map here

        # # Ensure that each individual sample (not the full batch) is yielded
        # for i in range(features.shape[0]):  # Iterate over the batch dimension
        #     feature = features[i]  # Get each individual feature of shape (8, 1944)
        #     label = labels[i]  # Corresponding label
        #     # print('feature_shape====================================================',features.shape)
        
        for feature, label in zip(features, labels):    
        # Yield each feature and label pair as you process them
        # for feature, label in zip(features, labels):
            if feature.shape != (8, 1944):
                raise ValueError(f"Expected feature shape (8, 1944), but got {feature.shape[-2:]}")
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
                tf.TensorSpec(shape=(8,1944), dtype=tf.float32),
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

# Example usage
features_dir = '/home/rag-tt/workspace/hog_features/train_features'
batch_size = 4
data_loader = create_data_loader(features_dir, batch_size)
print('data loader=',data_loader)

# Print a batch of the dataset
for batch_features, batch_labels in data_loader:
    print("Features:", batch_features.shape)
    print("Labels:", batch_labels)
    break