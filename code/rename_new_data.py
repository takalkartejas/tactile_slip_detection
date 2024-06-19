import os

# Assuming the folders are in the current working directory
base_path = '/home/rag-tt/tactile3/tactile_images'  # Change this to the actual path if needed

for i in range(2279, 0, -1):
    current_name = str(i)
    new_name = str(i + 2280)
    current_path = os.path.join(base_path, current_name)
    new_path = os.path.join(base_path, new_name)
    if os.path.exists(current_path):
        os.rename(current_path, new_path)
        print('Renamed', str(current_path),' to ', str(new_path))
    else:
        print('{current_path} does not exist')

print('Renaming complete')