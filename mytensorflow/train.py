import tensorflow as tf
import mytensorflow as mytf
from mytensorflow import utils


def get_sgd_train_op(loss, lr, lr_decay_factor=0.1, steps_per_decay=1e5):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(lr,
                                    global_step,
                                    steps_per_decay,
                                    lr_decay_factor,
                                    staircase=True)

    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = utils._add_loss_summaries(loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)

    grads = opt.compute_gradients(loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(
                            grads, global_step=global_step)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
                            mytf.MOVING_AVERAGE_DECAY_FOR_VARS,
                            global_step)
    variables_averages_op = variable_averages.apply(
                                tf.trainable_variables())

    # Tie the apply gradient operation and variable averaging into a
    # single operation.
    with tf.control_dependencies([apply_gradient_op,
                                  variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train(train_op, eval_fn, total_loss, num_steps, print_every_n=1e2,
          eval_every_n=1e2, summarize_every_n=1e2, diverge_threshold=1e3):
    init_op = tf.initialize_all_variables()
    summaries_op = tf.merge_all_summaries()
    with tf.Session() as sess:
        writer = tf.train.SummaryWriter("log", sess.graph_def)
        sess.run(init_op)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in xrange(int(num_steps)):
            if i % summarize_every_n == 0:
                _, summaries, loss_value = sess.run(
                                        [train_op, summaries_op, total_loss],
                                        feed_dict=utils._get_train_feed_dict())
                writer.add_summary(summaries, i)
            else:
                _, loss_value = sess.run([train_op, total_loss],
                                         feed_dict=utils._get_train_feed_dict())

            if loss_value > diverge_threshold:
                print('Training is diverging. Exiting.')
                return

            if i % print_every_n == 0:
                print "Loss @ iteration %05d: %f" % (i, loss_value)

            if i % eval_every_n == 0 and i > 0:
                print eval_fn()

        coord.request_stop()
        coord.join(threads)
