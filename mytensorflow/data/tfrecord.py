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
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
                           'height': _int64_feature(rows),
                           'width': _int64_feature(cols),
                           'depth': _int64_feature(depth),
                           'label': _int64_feature(int(label)),
                           'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())


def fixed_size_convert_to(root, tfrecord_filename):
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    for images, labels in imgdir.generate_image_batches(root, batch_size=1):
        encode_image(writer, images[0], labels[0])


def resize_and_convert_to(root, size, tfrecord_filename):
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    for images, labels in imgdir.generate_image_batches(root, batch_size=1):
        image = cv2.resize(images[0], size)
        encode_image(writer, image, labels[0])


def convert_to(root, tfrecord_filename, size=None):
    if size is not None:
        resize_and_convert_to(root, size, tfrecord_filename)
    else:
        fixed_size_convert_to(root, tfrecord_filename)


# =============================================================================
#                         Decoding functions
# =============================================================================
def read_and_decode_image(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # TODO(kjchavez): Update this to use the new parse_single_example API from
    # release 0.7 of TensorFlow.
    features = tf.parse_single_example(
                   serialized_example,
                   dense_keys=["image_raw", "label", "depth", "width",
                               "height"],
                   dense_types=[tf.string, tf.int64, tf.int64, tf.int64,
                                tf.int64])

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    depth_major = tf.reshape(image, [features['depth'],
                                     features['height'],
                                     features['depth']])

    return tf.transpose(depth_major, [1, 2, 0]), features['label']
