"""
    A set of data serialization and deserialization utilities.
"""
import mytensorflow as mytf
import mytensorflow.data.tfrecord as mytf_tfrecord
import tensorflow as tf
import yaml


class TFRecordSource(object):
    def __init__(self, name, metadata_file, batch_size,
                 num_epochs=None, shuffle=False, whiten=False):
        with open(metadata_file) as fp:
            metadata = yaml.load(fp)

        assert 'num_classes' in metadata
        assert 'train' in metadata
        assert 'test' in metadata
        assert 'shape' in metadata

        self.num_classes = metadata['num_classes']
        self.img_shape = metadata['shape']
        self.num_test_examples = metadata['num_test_examples']
        self.num_train_examples = metadata['num_train_examples']
        self.batch_size = batch_size
        shape = metadata['shape']
        train_filenames = metadata['train']
        test_filenames = metadata['test']

        with tf.name_scope(name + ".train"):
            filename_queue = tf.train.string_input_producer(train_filenames)

            # Even when reading in multiple threads, share the filename
            # queue.
            image, label = mytf_tfrecord.read_and_decode_image(
                                filename_queue, shape)

            # Whiten image.
            if whiten:
                image = tf.image.per_image_whitening(image)

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

            self.train_data = tf.cast(images, tf.float32)
            self.train_sparse_labels = tf.cast(sparse_labels, tf.int32)

            self.train_dense_labels = mytf.utils.encode_one_hot(self.train_sparse_labels,
                                                          self.num_classes)

        with tf.name_scope(name + ".test"):
            filename_queue = tf.train.string_input_producer(test_filenames)

            # Even when reading in multiple threads, share the filename
            # queue.
            image, label = mytf_tfrecord.read_and_decode_image(
                                filename_queue, shape)

            # Whiten image.
            if whiten:
                image = tf.image.per_image_whitening(image)

            images, sparse_labels = tf.train.batch(
                [image, label], batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size)

            self.test_data = tf.cast(images, tf.float32)
            self.test_sparse_labels = tf.cast(sparse_labels, tf.int32)

            self.test_dense_labels = mytf.utils.encode_one_hot(self.test_sparse_labels,
                                                          self.num_classes)

    def get_train_feed_dict(self):
        return {}

    def get_eval_feed_dict(self):
        return {}
