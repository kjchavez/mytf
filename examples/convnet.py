"""
Examples of using mytensorflow to build models.
"""
import mytensorflow as mytf
import tensorflow as tf

# Convolutional Neural Networks.
BATCH_SIZE = 64
dataset = mytf.source.TFRecordSource('dataset', 'housemates.metadata',
                                     BATCH_SIZE)
conv1 = mytf.conv_layer_with_l2_reg('conv1', dataset.data, 16, 0)
conv2 = mytf.conv_layer_with_l2_reg('conv2', conv1.activations, 32, 0)
fc3 = mytf.fc_layer_with_l2_reg('fc3', conv2.activations, dataset.num_classes,
                                0)

tf.add_to_collection('train_feed_dict', ('abc', 4))
# Loss (total_loss = cross_entropy_)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    fc3.output, dataset.dense_labels,
                    name='cross_entropy_per_example')

cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
tf.add_to_collection('losses', cross_entropy_mean)

total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

components = [conv1, conv2, fc3]
t = mytf.trainer.SGDTrainer(total_loss, components, 1e-5)
t.run(1000, print_every_n=1, summarize_every_n=10)

print tf.get_collection('train_feed_dict')
