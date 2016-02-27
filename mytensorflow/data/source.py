"""
    A set of data serialization and deserialization utilities.
"""

import tensorflow as tf


def read_and_decode_image_tfrecord(filename_queue):
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


def encode_one_hot(labels, num_classes):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, num_classes]), 1.0, 0.0)
    return onehot_labels


class TFRecordSource(object):
    def __init__(self, name, filenames, batch_size,
                 num_classes, num_epochs=None, shuffle=False):
        if isinstance(filenames, (str, unicode)):
            filenames = [filenames]

        with tf.name_scope(name):
            filename_queue = tf.train.string_input_producer(filenames)

            # Even when reading in multiple threads, share the filename
            # queue.
            image, label = read_and_decode_image_tfrecord(filename_queue)

            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            if shuffle:
                images, sparse_labels = tf.train.shuffle_batch(
                    [image, label], batch_size=batch_size, num_threads=2,
                    capacity=1000 + 3 * batch_size,
                    # Ensures a minimum amount of shuffling of examples.
                    min_after_dequeue=1000)
            else:
                images, sparse_labels = tf.train.batch(
                    [image, label], batch_size=batch_size, num_threads=2,
                    capacity=1000 + 3 * batch_size)

            self.data = tf.cast(images, tf.float32)
            self.sparse_labels = tf.cast(sparse_labels, tf.int32)

            # TODO(kjchavez): num_classes should come from the data source
            # itself if possible.
            self.dense_labels = encode_one_hot(self.sparse_labels, num_classes)
