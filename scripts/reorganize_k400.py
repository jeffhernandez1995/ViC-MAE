from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm

# Path to the main dataset directory
dataset_path = Path('/scratch/jeh16/datasets/k400/')

# The subdirectories for the video data
video_subdirs = ['train', 'test', 'val']

# The CSV files with the annotations
csv_files = ['train.csv', 'test.csv', 'val.csv']

# Go through each CSV file
for csv_file, video_subdir in zip(csv_files, video_subdirs):
    print(f"Processing {video_subdir} videos...")

    # Read the CSV file
    df = pd.read_csv(dataset_path / 'annotations' / csv_file)

    # Create a progress bar
    pbar = tqdm(total=df.shape[0])

    # Go through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the class label and clean it
        class_label = row['label']
        class_label = ''.join(e for e in class_label if e.isalnum()).replace(' ', '_')
        
        # Get the video file name
        video_file = f"{row['youtube_id']}_{str(row['time_start']).zfill(6)}_{str(row['time_end']).zfill(6)}.mp4"
        
        # Create a directory path for the class within the appropriate subdir (train/test/val)
        class_dir_path = dataset_path / video_subdir / class_label

        # If the class directory doesn't exist, create it
        class_dir_path.mkdir(parents=True, exist_ok=True)

        # Define source and destination paths for the video file
        source_path = dataset_path / video_subdir / video_file
        dest_path = class_dir_path / video_file

        # If the video file exists in the source path, move it to the destination path
        if source_path.exists():
            # print(f"Moving {source_path} to {dest_path}")
            # assert 2 == 1
            shutil.move(str(source_path), str(dest_path))

        # Update the progress bar
        pbar.update()

    # Close the progress bar
    pbar.close()
