import tensorflow as tf


class Training:

    def __init__(self, config):
        self.config = config
        pass

    def loss(self, logits, labels):
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('loss', cross_entropy_mean)
        tf.summary.scalar('loss', cross_entropy_mean)
        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('loss'), name='total_loss')

    # def loss(self, logits, labels):
    #     """Add L2Loss to all the trainable variables.
    #     Add summary for for "Loss" and "Loss/avg".
    #     Args:
    #       logits: Logits from inference().
    #       labels: Labels from distorted_inputs or inputs(). 1-D tensor
    #               of shape [batch_size]
    #     Returns:
    #       Loss tensor of type float.
    #     """
    #     # Reshape the labels into a dense Tensor of
    #     # shape [batch_size, NUM_CLASSES].
    #     sparse_labels = tf.reshape(labels, [self.config.BATCH_SIZE, 1])
    #     indices = tf.reshape(tf.range(0, self.config.BATCH_SIZE), [self.config.BATCH_SIZE, 1])
    #     concated = tf.concat([indices, sparse_labels], 1)
    #     dense_labels = tf.sparse_to_dense(concated, [self.config.BATCH_SIZE, self.config.NUM_CLASSES], 1.0, 0.0)
    #
    #     # Calculate the average cross entropy loss across the batch.
    #     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=dense_labels, name='cross_entropy_per_example')
    #     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    #     tf.add_to_collection('losses', cross_entropy_mean)
    #
    #     # The total loss is defined as the cross entropy loss plus all of the weight
    #     # decay terms (L2 loss).
    #     return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def accuracy(self, logits, labels):
        # with tf.name_scope('correct_prediction'):
        #     # s = labels.get_shape()
        #     # s1 = logits.get_shape()
        #     # labels = tf.reshape(labels, (logits.shape[0].value, logits.shape[1].value))
        #     # labels.set_shape((logits.shape[0].value, logits.shape[1].value))
        #     correct_prediction = tf.equal(tf.argmax(labels, axis=0), tf.argmax(logits, axis=1))
        # with tf.name_scope('accuracy'):
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #     tf.add_to_collection('accuracy', accuracy)
        # tf.summary.scalar('accuracy', accuracy)
        # return tf.add_n(tf.get_collection('accuracy'), name='accuracy')

        """Evaluate the quality of the logits at predicting the label.
          Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size], with values in the
              range [0, NUM_CLASSES).
          Returns:
            A scalar int32 tensor with the number of examples (out of batch_size)
            that were predicted correctly.
          """
        #print('Evaluation..')
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label's is was in the top k (here k=1)
        # of all logits for that example.
        with tf.name_scope('correct_prediction'):
            correct = tf.nn.in_top_k(logits, labels, 1)
            num_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        with tf.name_scope('accuracy'):
            acc_percent = (num_correct / self.config.BATCH_SIZE) * 100
        tf.add_to_collection('accuracy', acc_percent)
        tf.summary.scalar('accuracy', acc_percent)
        return tf.add_n(tf.get_collection('accuracy'), name='accuracy')

    def train(self, loss, global_step, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN):
        # Variables that affect learning rate.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / self.config.BATCH_SIZE
        decay_steps = int(num_batches_per_epoch * self.config.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.config.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        self.config.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        #loss_averages_op = self._add_loss_summaries(total_loss)

        # Compute gradients.
        ##with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        # grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         tf.summary.histogram(var.op.name + '/gradients', grad)

        optimizer = tf.train.AdamOptimizer(lr)
        # optimizer = tf.train.GradientDescentOptimizer(lr)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(self.config.MOVING_AVERAGE_DECAY, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())
        #
        # with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        #     train_op = tf.no_op(name='train')

        return train_op

    # def _add_loss_summaries(self, total_loss):
    #     # Compute the moving average of all individual losses and the total loss.
    #     loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    #     losses = tf.get_collection('losses')
    #     loss_averages_op = loss_averages.apply(losses + [total_loss])
    #
    #     # Attach a scalar summary to all individual losses and the total loss; do the
    #     # same for the averaged version of the losses.
    #     for l in losses + [total_loss]:
    #         # Name each loss as '(raw)' and name the moving average version of the loss
    #         # as the original loss name.
    #         tf.summary.scalar(l.op.name + ' (raw)', l)
    #         tf.summary.scalar(l.op.name, loss_averages.average(l))
    #
    #     return loss_averages_op
