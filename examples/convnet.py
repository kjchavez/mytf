"""
Examples of using mytensorflow to build models.
"""
import mytensorflow as mytf
import tensorflow as tf

# Convolutional Neural Networks.
BATCH_SIZE = 64
dataset = mytf.source.TFRecordSource('dataset', 'housemates.metadata',
                                     BATCH_SIZE, whiten=True)

c1 = mytf.ConvLayer("conv1", dataset.img_shape[2], 32)
c2 = mytf.ConvLayer("conv2", c1.depth, 64)
fc3 = mytf.FullyConnectedLayer("fc3",
                               dataset.img_shape[0]*dataset.img_shape[1]*64/4,
                               512, keep_prob=1.0)
fc4 = mytf.FullyConnectedLayer("fc4", 512, dataset.num_classes, keep_prob=1.)

def inference(X):
    output = c1.transform(X)
    output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    output = c2.transform(output)
    output = fc3.transform(output)
    output = fc4.transform(output)
    return output

def true_count(logits, labels):
    """ Determines accuracy of prediction. """
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    return count

# Training loss.
logits = inference(dataset.train_data)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits, dataset.train_dense_labels,
                    name='cross_entropy')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
tf.add_to_collection('losses', cross_entropy_mean)
print tf.get_collection('losses')
total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')


true_count = true_count(inference(dataset.test_data),
                        dataset.test_dense_labels)
def evaluate():
    """ Should only be called inside a tf.Session().
    Returns a formatted string to be displayed as evaluation results.
    """
    num_correct = 0
    total = 0
    for i in xrange(dataset.num_test_examples // dataset.batch_size):
        num_correct += true_count.eval(feed_dict=mytf.utils._get_eval_feed_dict())
        total += dataset.batch_size

    return "Accuracy on validation set: %0.3f%%" % (100*float(num_correct) /
                                                     total)

train_op = mytf.train.get_sgd_train_op(total_loss, lr=1e-5,
                                       steps_per_decay=1e3)
mytf.train.train(train_op, evaluate, total_loss, 1e4)
