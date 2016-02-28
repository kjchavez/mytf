"""
    A set of data serialization and deserialization utilities.
"""
import mytensorflow as mytf
import tensorflow as tf


class TFRecordSource(object):
    def __init__(self, name, filenames, batch_size,
                 num_classes, num_epochs=None, shuffle=False):
        self.num_classes = num_classes
        if isinstance(filenames, (str, unicode)):
            filenames = [filenames]

        with tf.name_scope(name):
            filename_queue = tf.train.string_input_producer(filenames)

            # Even when reading in multiple threads, share the filename
            # queue.
            image, label = mytf.data.tfrecord.read_and_decode_image(
                                filename_queue)

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
            self.dense_labels = mytf.utils.encode_one_hot(self.sparse_labels,
                                                          num_classes)
