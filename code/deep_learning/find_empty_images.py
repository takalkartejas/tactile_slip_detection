import os

def find_empty_images(root_dir):
    empty_images = []
    
    # Loop through each subdirectory from 1 to 20000
    for subdir in range(0, 28589):
        folder_path = os.path.join(root_dir, str(subdir))
        
        # Check if subdirectory exists
        if not os.path.exists(folder_path):
            continue
        
        # Loop through each file in the subdirectory
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Check if it's a file and has 0 bytes (empty file)
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                empty_images.append(file_path)

    return empty_images


# Usage example:
root_directory = '/home/rag-tt/workspace/train_data/'  # Replace with your root folder path
empty_images = find_empty_images(root_directory)

# Print the list of empty images
if empty_images:
    print(f"Found {len(empty_images)} empty images:")
    for img in empty_images:
        print(img)
else:
    print("No empty images found.")