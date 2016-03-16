from __future__ import print_function

import tensorflow as tf
import mytensorflow as mytf
import mytensorflow.utils as utils


class SGDTrainer(object):
    """Train the model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    """
    def __init__(self, total_loss, eval_accuracy, dataset, components, lr,
                 lr_decay_factor=0.1,
                 steps_per_decay=1e5):
        # Decay the learning rate exponentially based on the number of steps.
        self.global_step = tf.Variable(0, trainable=False)
        self.total_loss = total_loss

        lr = tf.train.exponential_decay(lr,
                                        self.global_step,
                                        steps_per_decay,
                                        lr_decay_factor,
                                        staircase=True)

        tf.scalar_summary('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = utils._add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(
                                grads, global_step=self.global_step)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
                                mytf.MOVING_AVERAGE_DECAY_FOR_VARS,
                                self.global_step)
        variables_averages_op = variable_averages.apply(
                                    tf.trainable_variables())

        # Tie the apply gradient operation and variable averaging into a
        # single operation.
        with tf.control_dependencies([apply_gradient_op,
                                      variables_averages_op]):
            self.train_op = tf.no_op(name='train')

        # An operator to initialize all variables.
        self.init = tf.initialize_all_variables()

        # Feed dicts needed for train/eval settings.
        # TODO(kjchavez): It's odd that I need to pass in the components, just
        # to do this. Seems like I could use the same tf.collection() approach
        # and keep track of these dicts at a global level?
        self.eval_accuracy = eval_accuracy
        self.components = components
        self.summaries = tf.merge_all_summaries()
        self.dataset = dataset

    def evaluate(self):
        accuracy = 0
        num_steps = self.dataset.num_test_examples // self.dataset.batch_size
        for i in xrange(num_steps):
            accuracy += self.eval_accuracy.eval(
                feed_dict=utils._get_eval_feed_dict(self.components))

        return accuracy / num_steps

    def run(self, iterations, print_every_n=1000, eval_every_n=10000,
            summarize_every_n=1000, diverge_threshold=1e3):
        with tf.Session() as sess:
            writer = tf.train.SummaryWriter("log", sess.graph_def)
            sess.run(self.init)
            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in xrange(iterations):
                train_feed_dict = utils._get_train_feed_dict(self.components)
                if i % summarize_every_n == 0:
                    _, summaries, loss = sess.run(
                                            [self.train_op, self.summaries,
                                             self.total_loss],
                                            feed_dict=train_feed_dict)
                    writer.add_summary(summaries, i)
                else:
                    _, loss = sess.run([self.train_op, self.total_loss],
                                       feed_dict=train_feed_dict)

                if i % print_every_n == 0:
                    print('Iter %d: loss = %0.5f' % (i, loss))

                if loss > diverge_threshold:
                    print('Training is diverging. Exiting.')
                    return

                if i % eval_every_n == 0 and i != 0:
                    accuracy = self.evaluate()
                    print ('Accuracy on validation set: %0.5f' % accuracy)

            coord.request_stop()
            coord.join(threads)
