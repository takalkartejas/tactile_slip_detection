import os
import pandas as pd

def delete_small_csv_files(directory, min_rows=29):
    try:
        # List all files in the directory
        files = os.listdir(directory)
        
        for file in files:
            # Construct full file path
            file_path = os.path.join(directory, file)
            
            # Check if it's a CSV file
            if file.endswith('.csv'):
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Check the number of rows
                    if len(df) < min_rows:
                        # Delete the file
                        os.remove(file_path)
                        print(f"Deleted {file_path} (less than {min_rows} rows)")
                except pd.errors.EmptyDataError:
                    # Handle the case where the CSV file is empty
                    os.remove(file_path)
                    print(f"Deleted {file_path} (empty file)")
                except Exception as e:
                    print(f"Could not process file {file_path}: {e}")
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access '{directory}'.")

# Example usage
directory_to_check = os.path.dirname(os.path.abspath(__file__))
delete_small_csv_files(directory_to_check)