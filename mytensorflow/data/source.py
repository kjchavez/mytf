"""
    A set of data serialization and deserialization utilities.
"""
import mytensorflow as mytf
import mytensorflow.data.tfrecord as mytf_tfrecord
import tensorflow as tf
import yaml


class TFRecordSource(object):
    def __init__(self, name, metadata_file, batch_size,
                 num_epochs=None, shuffle=False):
        with open(metadata_file) as fp:
            metadata = yaml.load(fp)

        assert 'num_classes' in metadata
        assert 'filenames' in metadata
        assert 'shape' in metadata

        self.num_classes = metadata['num_classes']
        shape = metadata['shape']
        filenames = metadata['filenames']

        with tf.name_scope(name):
            filename_queue = tf.train.string_input_producer(filenames)

            # Even when reading in multiple threads, share the filename
            # queue.
            image, label = mytf_tfrecord.read_and_decode_image(
                                filename_queue, shape)

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
                                                          self.num_classes)
