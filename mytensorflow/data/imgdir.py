"""
    Utilities for working for image data stored as a directory with one sub-
    directory per class.

    Makes a couple of assumptions about the data:

    1. The list of all filenames with their associate folder fits in memory.
       Not the file contents, just the names.
    2. The image can be read via cv2.imread().
    3. All images are stored in the same format (jpg, png, etc.). This
       assumption can be relaxed if needed in the near future.
"""
from __future__ import print_function
import os
import glob
import numpy as np
import cv2


def generate_label_mapping(root_directory, output_file):
    """ Generates a mapping of folder name to an integer index.

    Writes this to mapping to 'output_file' and returns a dictionary mapping
    folder name to index.
    """
    mapping = {}
    label_index = 0
    with open(output_file, 'w') as fp:
        for folder in os.listdir(root_directory):
            if os.path.isdir(os.path.join(root_directory, folder)):
                print(folder, label_index)
                print(folder, label_index, file=fp)
                mapping[folder] = label_index
                label_index += 1

    return mapping


def get_filenames_with_folder(root, ext="jpg"):
    """ Returns a list of tuples (filename, foldername) for all files that are
    within a subdirectory of 'root' with the given extension.
    """
    tuples = []
    for folder in os.listdir(root):
        path = os.path.join(root, folder)
        for filename in glob.glob(os.path.join(path, "*.%s" % ext)):
            tuples.append((filename, folder))

    return tuples


def generate_image_batches(root, ext="jpg", batch_size=100, shuffle=True):
    """ A generator yielding a list of images and a list of corresponding labels.

    Args:
        root (string):        the root path for the image directory
        label_mapping (dict): a dictionary mapping folder name to class index
        ext (string):         the image-specific extension of files in the
                              image directory.
        batch_size (int):     number of images, labels that will be returned at
                              a time. Note this many images *must* fit in RAM
        shuffle (bool):       if true, data points will be returned in a
                              shuffled order.

    Yields:
        (list of images, list of labels)

    """
    label_mapping_filename = "%s.labels" % os.path.basename(root)
    label_mapping = generate_label_mapping(root, label_mapping_filename)
    image_files, labels = zip(*get_filenames_with_folder(root, ext=ext))
    image_files = np.array(image_files)
    labels = np.array([label_mapping[x] for x in labels], dtype=np.uint32)
    N = len(labels)  # Total number of images
    if shuffle:
        order = np.arange(N)
        np.random.shuffle(order)
        image_files = image_files[order]
        labels = labels[order]

    for i in xrange(0, N, batch_size):
        num_examples = min(batch_size, N - i)
        batch_files = image_files[i:i+num_examples]
        batch_labels = labels[i:i+num_examples]
        batch_images = [cv2.imread(fname) for fname in batch_files]
        if i % 500 == 0:
            print('Processing images (%d / % d)' % (i, N))

        yield batch_images, batch_labels
