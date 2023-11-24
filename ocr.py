#!/usr/bin/env python
from PIL import Image
import numpy as np
import os

DATA_DIR = 'C:/Users/RhyzenWorkspace/OneDrive/ocr/data/'
TEST_DIR = 'C:/Users/RhyzenWorkspace/OneDrive/ocr/test/'
DATASET = 'E-13B/'

TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels.idx1-ubyte'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

def read_image(path):
    return np.asarray(Image.open(path).convert('L'))

def read_test_images(folder_path):
    images = []
    print(f"Reading test images from: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return images

    for filename in sorted(os.listdir(folder_path)):
        print(f"Found file: {filename}")
        if filename.lower().endswith('.png'):
            path = os.path.join(folder_path, filename)
            image = np.asarray(Image.open(path).convert('L'))
            images.append(image)
        else:
            print(f"Skipping non-PNG file: {filename}")

    return images

def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels

def extract_label_from_filename(filename):
    """
    Extracts the label from the filename.
    Assumes filenames are like '0.png', '1.png', etc.
    """
    label_part = filename.split('.')[0]
    if label_part.isdigit():
        return int(label_part)
    else:
        print(f"Skipping file with unrecognized name: {filename}")
        return None  # Return None to indicate an invalid or unrecognized label

def read_custom_images_labels(folder_path):
    images = []
    labels = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            path = os.path.join(folder_path, filename)
            image = np.asarray(Image.open(path).convert('L'))
            label = extract_label_from_filename(filename)
            if label is not None:
                images.append(image)
                labels.append(label)
    return images, labels

def read_images_from_folders(root_folder):
    images = []
    labels = []
    for folder_name in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.png'):
                    path = os.path.join(folder_path, filename)
                    image = np.asarray(Image.open(path).convert('L'))
                    images.append(image)
                    labels.append(int(folder_name))  # Assuming folder names are integers
    return images, labels

def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def dist(x, y):
    """
    Returns the Euclidean distance between vectors `x` and `y`.
    """
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
            for x_i, y_i in zip(x, y)
        ]
    ) ** (0.5)


def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    if not l:
        return None  # or a default value, or raise an exception
    return max(l, key=l.count)


def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_sample_idx, test_sample in enumerate(X_test):
        print(test_sample_idx, end=' ', flush=True)
        training_distances = get_training_distances_for_test_sample(X_train, test_sample)
        sorted_distance_indices = [pair[0] for pair in sorted(enumerate(training_distances), key=lambda x: x[1])]
        
        candidates = [y_train[idx] for idx in sorted_distance_indices[:k]]

        if not candidates:  # Check if candidates list is empty
            print(f"No candidates found for test sample {test_sample_idx}, assigning default label.")
            y_pred.append(0)  # Assign a default label or handle it as needed
            continue

        top_candidate = get_most_frequent_element(candidates)
        y_pred.append(top_candidate)
    print()
    return y_pred

# def main():
#     n_train = 10000
#     n_test = 1
#     k = 7
#     print(f'Dataset: {DATASET}')
#     print(f'n_train: {n_train}')
#     print(f'n_test: {n_test}')
#     print(f'k: {k}')

#     custom_data_folder = 'C:/Users/Rhyzen/OneDrive/ocr/data/E-13B/'

#     # X_train = read_images(TRAIN_DATA_FILENAME, n_train)
#     # y_train = read_labels(TRAIN_LABELS_FILENAME, n_train)
#     # X_test = read_images(TEST_DATA_FILENAME, n_test)
#     X_train, y_train = read_custom_images_labels(custom_data_folder)
#     X_test = [read_image(f'{TEST_DIR}our_test.bmp')]
#     y_test = read_labels(TEST_LABELS_FILENAME, n_test)

#     X_train = extract_features(X_train)
#     X_test = extract_features(X_test)

#     y_pred = knn(X_train, y_train, X_test, k)

#     accuracy = sum([
#         int(y_pred_i == y_test_i)
#         for y_pred_i, y_test_i
#         in zip(y_pred, y_test)
#     ]) / len(y_test)
    
#     print(f'Predicted labels: {y_pred}')

#     print(f'Accuracy: {accuracy * 100}%')

def main():
    k = 10
    print(f'Dataset: Custom')
    print(f'k: {k}')
    
    # Reading training data from folder s
    training_data_folder = f'{DATA_DIR}{DATASET}'  # Update this with your folder path
    print(f"Loading training data from {training_data_folder}")
    X_train, y_train = read_images_from_folders(training_data_folder)
    print(f"Loaded {len(X_train)} training images")
    print(f"First few training labels: {y_train[:5]}")

    if not X_train:
        print("No training data loaded. Check your file paths and folder structure.")
        return

    X_train = extract_features(X_train)

    # Reading test data directly from folder (without
    #  subfolders)
    test_data_folder = f'{TEST_DIR}'  # Update this with your folder path
    print(f"Loading test data from {test_data_folder}")
    X_test = read_test_images(test_data_folder)
    print(f"Loaded {len(X_test)} test images")

    if not X_test:
        print("No test data loaded. Check your file paths and folder structure.")
        return

    X_test = extract_features(X_test)

    # Predict labels for test data
    print("Predicting labels for test data...")
    y_pred = knn(X_train, y_train, X_test, k)

    if not y_pred:
        print("No predictions were made. Check if your KNN function is working correctly.")
        return

    # Concatenating predicted labels to form a password
    password = ''.join(map(str, y_pred))
    print(f'\nPassword is: {password}')

if __name__ == '__main__':
    main()


# Reference:
# Clumsy Computer. (2020, July 7). Coding OCR with machine learning from scratch in Python — no libraries or imports! (From Scratch #2). Read on November 14, 2023. (https://git.sr.ht/~vladh/clumsycomputer/tree/main/item/from-scratch-2-ocr)
# Code optimization: ChatGPT, Me