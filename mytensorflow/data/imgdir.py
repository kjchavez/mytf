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

def create_label_mapping(root_directory):
    """ Generates a mapping of folder name to an integer index.

    Returns a dictionary mapping folder name to index.
    """
    mapping = {}
    label_index = 0
    for folder in os.listdir(root_directory):
        if os.path.isdir(os.path.join(root_directory, folder)):
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


class ImageDirectory(object):
    def __init__(self, root, ext="jpg"):
        self.root = root
        self.mapping = create_label_mapping(root)
        self.data_points = get_filenames_with_folder(root, ext=ext)

    def train_test_split(self, train=0.8):
        """ Shuffles the data points and returns list of training data and
        testing data.
        """
        N = len(self.data_points)
        order = np.arange(N)
        np.random.shuffle(order)
        train_split = [(self.data_points[order[i]][0],
                        self.mapping[self.data_points[order[i]][1]]) for i in
                       xrange(int(0.8*N))]
        test_split = [(self.data_points[order[i]][0],
                        self.mapping[self.data_points[order[i]][1]]) for i in
                       xrange(int(0.8*N)+1, N)]

        return train_split, test_split

    def num_classes(self):
        return len(self.mapping)

    def num_images(self):
        return len(self.data_points)

