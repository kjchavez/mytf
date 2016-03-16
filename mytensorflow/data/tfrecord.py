""" Utilities for working with TFRecords. """
from __future__ import print_function

import tensorflow as tf
import cv2
from mytensorflow.data import imgdir


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# =============================================================================
#                         Encoding functions
# =============================================================================
def encode_image(writer, image, label):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(int(label)),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())


# =============================================================================
#                         Decoding functions
# =============================================================================
# TODO(kjchavez): Figure out how to reconcile dynamically sized images, the
# code below doesn't work.
def read_and_decode_image(filename_queue, shape):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    row_major = tf.reshape(image, shape)

    return row_major, features['label']
