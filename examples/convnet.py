"""
Examples of using mytensorflow to build models.
"""
import mytensorflow as mytf
import tensorflow as tf

# Convolutional Neural Networks.
BATCH_SIZE = 4
dataset = mytf.source.TFRecordSource('dataset', 'housemates.metadata', BATCH_SIZE)
conv1 = mytf.conv_layer_with_l2_reg('conv1', dataset.data, 16, 1e-4)
conv2 = mytf.conv_layer_with_l2_reg('conv2', conv1.activations, 16, 1e-4)
fc3 = mytf.fc_layer_with_l2_reg('fc3', conv2.activations, dataset.num_classes, 1e-4)
