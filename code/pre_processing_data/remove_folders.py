import os
import shutil

def remove_folders_in_range(parent_dir, min_val, max_val):
    """
    Remove subdirectories within a specified range in a given parent directory.

    :param parent_dir: The path to the parent directory containing numbered subdirectories.
    :param min_val: The minimum value of the subdirectory number to be removed.
    :param max_val: The maximum value of the subdirectory number to be removed.
    """
    if not os.path.exists(parent_dir):
        raise ValueError(f"The directory {parent_dir} does not exist.")

    for folder_name in os.listdir(parent_dir):
        try:
            folder_num = int(folder_name)
            if min_val <= folder_num <= max_val:
                folder_path = os.path.join(parent_dir, folder_name)
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Removed directory: {folder_path}")
        except ValueError:
            # Skip non-numeric directories
            continue

# Example usage:
parent_directory = '/home/rag-tt/train_data'
minimum_value = 5590
maximum_value = 10000

remove_folders_in_range(parent_directory, minimum_value, maximum_value)