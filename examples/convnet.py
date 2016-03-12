"""
Examples of using mytensorflow to build models.
"""
import mytensorflow as mytf
import tensorflow as tf

# Convolutional Neural Networks.
BATCH_SIZE = 128
dataset = mytf.source.TFRecordSource('dataset', 'minihousemates.metadata',
                                     BATCH_SIZE)

# Roughly zero mean:
X = tf.image.per_image_whitening(dataset.data)
conv1 = mytf.conv_layer_with_l2_reg('conv1', X, 64, 0)
pool1 = tf.nn.max_pool(conv1.activations, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
conv2 = mytf.conv_layer_with_l2_reg('conv2', pool1, 64, 0)
pool2 = tf.nn.max_pool(conv2.activations, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
fc3 = mytf.fc_layer_with_l2_reg('fc3', conv2.activations, 256, 0, keep_prob=1,
        W_init=tf.truncated_normal_initializer(stddev=1e-2))
fc4 = mytf.fc_layer_with_l2_reg('fc4', fc3.output, 128,
                                0, keep_prob=1)

fc5 = mytf.fc_layer('fc5', fc4.output, dataset.num_classes, keep_prob=1)

# tf.add_to_collection('train_feed_dict', ('abc', 4))
# Loss (total_loss = cross_entropy_)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    fc5.output, dataset.dense_labels,
                    name='cross_entropy_per_example')

cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
tf.add_to_collection('losses', cross_entropy_mean)

total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

components = [conv1, conv2, fc3, fc4, fc5]
t = mytf.trainer.SGDTrainer(total_loss, components, 1e-1)
t.run(10000, print_every_n=10, summarize_every_n=10)

# print tf.get_collection('train_feed_dict')
