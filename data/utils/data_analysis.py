import os
import random
import shutil
import pandas as pd

def split_data(label_file, train_dir, test_dir, test_ratio=0.1):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    df = pd.read_csv(label_file)

    unique_labels = df['landmark_id_transformed'].unique()
    for label in unique_labels:
        label_data = df[df['landmark_id_transformed'] == label]
        num_samples = int(len(label_data) * test_ratio)

        print(f"Label: {label}, Number of samples: {len(label_data)}, randonly selected {num_samples} test samples.")

        # Randomly select the samples to move
        test_samples = label_data.sample(n=num_samples)

        # Move the selected samples from train_dir to test_dir
        for _, row in test_samples.iterrows():
            file_id = row['id']
            file_path = os.path.join(train_dir, f"{file_id}.jpg")
            new_file_path = os.path.join(test_dir, f"{file_id}.jpg")
            shutil.move(file_path, new_file_path)

    print("Data split completed successfully.")



def count_files(directory):
    file_count = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    print(f"Number of files in {directory}: {file_count}")


if __name__ == '__main__':
    directory = '/Users/ruipenghan/Desktop/Academics/11. SP 2024/CS 444/project/data/small_data/train'
    count_files(directory)

    directory = '/Users/ruipenghan/Desktop/Academics/11. SP 2024/CS 444/project/data/small_data/test'
    count_files(directory)
    
    # label_file = '/Users/ruipenghan/Desktop/Academics/11. SP 2024/CS 444/project/data/small_data/small_data.csv'
    # train_dir = '/Users/ruipenghan/Desktop/Academics/11. SP 2024/CS 444/project/data/small_data/train'
    # test_dir = '/Users/ruipenghan/Desktop/Academics/11. SP 2024/CS 444/project/data/small_data/test'
    # split_data(label_file, train_dir, test_dir)
