import tensorflow as tf
from tensorlayer import activation


class Model:
    def __init__(self, config):
        self.config = config
        pass

    def inference_(self, x, mode_name):
        with tf.name_scope('inputs'):
            # image = tf.reshape(x, [-1, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.CHANNELS])
            # tf.summary.image('example_{}'.format(mode_name), image, 4)
            x = tf.reshape(x, [-1, self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.CHANNELS])

        with tf.name_scope('convolution_1'):
            convolution_1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[5, 5], padding="same", activation=activation.lrelu)

        with tf.name_scope('max_pooling_1'):
            max_pooling_1 = tf.layers.max_pooling2d(inputs=convolution_1, pool_size=[2, 2], strides=2)

        with tf.name_scope('convolution_2'):
            convolution_2 = tf.layers.conv2d(inputs=max_pooling_1, filters=128, kernel_size=[5, 5], padding="same", activation=activation.lrelu)

        with tf.name_scope('max_pooling_2'):
            max_pooling_2 = tf.layers.max_pooling2d(inputs=convolution_2, pool_size=[2, 2], strides=2)

        with tf.name_scope('convolution_3'):
            convolution_3 = tf.layers.conv2d(inputs=max_pooling_2, filters=256, kernel_size=[5, 5], padding="same", activation=activation.lrelu)

        with tf.name_scope('max_pooling_3'):
            max_pooling_3 = tf.layers.max_pooling2d(inputs=convolution_3, pool_size=[2, 2], strides=2)

        with tf.name_scope('dense_1'):
            max_pooling_3 = tf.reshape(max_pooling_3, [max_pooling_3.shape[0].value, max_pooling_3.shape[1].value * max_pooling_3.shape[2].value * max_pooling_3.shape[3].value])
            dense_1 = tf.layers.dense(inputs=max_pooling_3, units=2048, activation=activation.lrelu)

        with tf.name_scope('logits'):
            logits = tf.layers.dense(inputs=dense_1, units=self.config.NUM_CLASSES, activation=activation.lrelu)

        tf.summary.histogram('raw_logits_{}'.format(mode_name), logits)
        tf.summary.histogram('classes_prob_distribution_prediction_{}'.format(mode_name), tf.nn.softmax(logits=logits))

        return logits

