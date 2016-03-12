"""
Examples of using mytensorflow to build models.
"""
import mytensorflow as mytf
import tensorflow as tf

# Two layer neural network.
BATCH_SIZE = 64
dataset = mytf.source.TFRecordSource('dataset', 'minihousemates.metadata',
                                     BATCH_SIZE)

# Roughly zero mean:
X = (1.0 / 255.0) * (dataset.data - 127.0)
fc3 = mytf.fc_layer_with_l2_reg('fc3', X, 256, 0, keep_prob=1,
        W_init=tf.truncated_normal_initializer(stddev=4e-2))
fc4 = mytf.fc_layer_with_l2_reg('fc4', fc3.output, dataset.num_classes,
                                0, keep_prob=1,
        W_init=tf.truncated_normal_initializer(stddev=4e-2))

# Loss (total_loss = cross_entropy_)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    fc4.output, dataset.dense_labels,
                    name='cross_entropy_per_example')

cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
tf.add_to_collection('losses', cross_entropy_mean)

total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

components = [fc3, fc4]
print ('Training on small dataset. Should heavily overfit.')
t = mytf.trainer.SGDTrainer(total_loss, components, 1e-3)
t.run(10000, print_every_n=100, summarize_every_n=100)

