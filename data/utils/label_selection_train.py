import pandas as pd
import os
import shutil 
import glob


def select_train_labels(n=50, full_csv_path="", downloaded_train=""):
    # TODO: When train data is downloaded, we need to get list of ids downloaded
    downloaded_img_ids = []
    for file_path in glob.glob(downloaded_train + '/*.jpg'):
        file_name = os.path.basename(file_path)
        downloaded_img_ids.append(os.path.splitext(file_name)[0])

    full_train_data = pd.read_csv(full_csv_path)

    # Select only downloaded img ids
    downloaded_train_data = full_train_data[full_train_data['id'].isin(downloaded_img_ids)]

    # Count and sort the downloaded training data
    label_counts = downloaded_train_data.groupby("landmark_id").size().sort_values(ascending=False)

    selected_labels = label_counts.head(n).index.tolist()

    selected_train_data = downloaded_train_data[downloaded_train_data['landmark_id'].isin(selected_labels)]
    selected_train_data.to_csv('selected_train.csv', index=False)
    
    labels_and_count = [(landmark_id, count) for landmark_id, count in label_counts.head(n).items()]

    print(f"Size of selected_train_data: {selected_train_data.shape[0]}")
    print("List of image IDs and their occurrences")
    print(labels_and_count)
    return selected_labels


"""
    Moves selected training images from the src_data_path to dest_data_path
"""
def gather_train_data(selected_train_data="", src_data_path="", dest_data_path=""):
    if not os.path.exists(dest_data_path):
        os.makedirs(dest_data_path)
    
    image_formats = ['.jpg', '.jpeg', '.png']
    image_file_paths = []
    
    # Gets list of img ids (names)
    train_data = pd.read_csv(selected_train_data)
    image_ids = set(train_data["id"].tolist())

    for root, dirs, files in os.walk(src_data_path):
        for file in files:
            if file.lower().endswith(tuple(image_formats)):
                file_name = os.path.splitext(file)[0]
                # Ensures that img is within selected pool
                if file_name in image_ids:
                    image_file_paths.append(os.path.join(root, file))
    
    print(f"List of selected train data: {image_file_paths}")

    for image_file_path in image_file_paths:
        # Move the image file to the destination path
        shutil.move(image_file_path, dest_data_path)


def info_downloaded_train_data(full_csv_path="", downloaded_train=""):
    downloaded_img_ids = []
    
    for root, dirs, files in os.walk(downloaded_train):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_name = os.path.splitext(file)[0]
                downloaded_img_ids.append(file_name)

    full_train_data = pd.read_csv(full_csv_path)

    # Select only downloaded img ids
    downloaded_train_data = full_train_data[full_train_data['id'].isin(downloaded_img_ids)]

    # Count and sort the downloaded training data
    label_counts = downloaded_train_data.groupby("landmark_id").size().sort_values(ascending=False)

    print(f"Number of unique landmarks in downloaded data: {len(label_counts)}")

    # Find the minimum number of unique landmark labels that cover at least 10% of the downloaded_train_data
    min_coverage = 0.1  # 10% coverage
    total_samples = downloaded_train_data.shape[0]
    min_labels = 1  # Start with 1 label

    while min_labels <= len(label_counts):
        coverage = downloaded_train_data[downloaded_train_data['landmark_id'].isin(label_counts.head(min_labels).index)].shape[0] / total_samples
        if coverage >= min_coverage:
            break
        min_labels += 1

    print(f"Minimum number of unique landmark labels to cover at least 10% of the downloaded_train_data: {min_labels}")
    # labels_and_count = [(landmark_id, count) for landmark_id, count in label_counts.head(min_labels).items()]


def main():

    # info_downloaded_train_data(full_csv_path="", downloaded_train="")

    # select_train_labels(n=50, full_csv_path="../labels/train.csv", downloaded_train="")
    
    # gather_train_data(selected_train_data="", src_data_path="/Users/ruipenghan/Desktop/Academics/11. SP 2024/CS 444/project/test", dest_data_path="out")

if __name__ == '__main__':
    main()