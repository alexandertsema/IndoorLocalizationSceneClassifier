import tensorflow as tf


class Evaluation:
    def __init__(self, config):
        self.config = config
        pass

    def validate(self):
        pass

    def test(self):
        pass

    def loss(self, predictions, labels, mode_name):  # Calculate the average cross entropy loss across the batch.
        one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.config.NUM_CLASSES)
        with tf.name_scope('loss_function_{}'.format(mode_name)):
            loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=predictions)

        tf.add_to_collection('loss_{}'.format(mode_name), loss)
        tf.summary.scalar('loss_{}'.format(mode_name), loss)

        return tf.add_n(tf.get_collection('loss_{}'.format(mode_name)), name='loss_{}'.format(mode_name))

    def accuracy(self, predictions, labels, mode_name):  # Calculate the average accuracy across the batch.
        with tf.name_scope('correct_prediction_{}'.format(mode_name)):
            correct = tf.nn.in_top_k(predictions, labels, 1)
            num_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        with tf.name_scope('accuracy_{}'.format(mode_name)):
            acc_percent = (num_correct / self.config.BATCH_SIZE) * 100

        tf.add_to_collection('accuracy_{}'.format(mode_name), acc_percent)
        tf.summary.scalar('accuracy_{}'.format(mode_name), acc_percent)

        return tf.add_n(tf.get_collection('accuracy_{}'.format(mode_name)), name='accuracy_{}'.format(mode_name))
