"""
Examples of using mytensorflow to build models.
"""
import mytensorflow as mytf
import tensorflow as tf

# Convolutional Neural Networks.
NUM_CLASSES = 10
BATCH_SIZE = 4
X = tf.random_normal((BATCH_SIZE, 32, 32, 3))
conv1 = mytf.conv_layer_with_l2_reg('conv1', X, 16, 1e-4)
conv2 = mytf.conv_layer_with_l2_reg('conv2', conv1.activations, 16, 1e-4)
fc3 = mytf.fc_layer_with_l2_reg('fc3', conv2.activations, NUM_CLASSES, 1e-4)
