import tensorflow as tf
from tensorlayer import activation

from visualization.visualization import Visualization


class Cnn:
    def __init__(self, config):
        self.config = config
        self.visualization = Visualization(self.config)
        pass

    def inference(self, x, mode_name):
        histogram_summary = False if mode_name == self.config.MODE.VALIDATION else True
        kernel_image_summary = False if mode_name == self.config.MODE.VALIDATION else True
        activation_image_summary = False if mode_name == self.config.MODE.VALIDATION else True

        with tf.name_scope('inputs'):
            x = tf.reshape(x, [-1, self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.CHANNELS])

        tf.summary.image("/inputs", x, max_outputs=4)

        with tf.variable_scope('convolution1'):
            convolution_1 = self.conv_layer(input_tensor=x,
                                            depth_in=self.config.IMAGE_SIZE.CHANNELS,
                                            depth_out=64,
                                            mode_name=mode_name,
                                            histogram_summary=histogram_summary,
                                            kernel_image_summary=kernel_image_summary,
                                            activation_image_summary=activation_image_summary)

        with tf.variable_scope('max_pooling1'):
            max_pooling_1 = tf.layers.max_pooling2d(inputs=convolution_1, pool_size=[2, 2], strides=2)

        with tf.variable_scope('convolution2'):
            convolution_2 = self.conv_layer(input_tensor=max_pooling_1,
                                            depth_in=64,
                                            depth_out=128,
                                            mode_name=mode_name,
                                            histogram_summary=histogram_summary,
                                            kernel_image_summary=False,
                                            activation_image_summary=activation_image_summary)

        with tf.variable_scope('max_pooling2'):
            max_pooling_2 = tf.layers.max_pooling2d(inputs=convolution_2, pool_size=[2, 2], strides=2)

        with tf.variable_scope('convolution3'):
            convolution_3 = self.conv_layer(input_tensor=max_pooling_2,
                                            depth_in=128,
                                            depth_out=256,
                                            mode_name=mode_name,
                                            histogram_summary=histogram_summary,
                                            kernel_image_summary=False,
                                            activation_image_summary=activation_image_summary)

        with tf.variable_scope('max_pooling3'):
            max_pooling_3 = tf.layers.max_pooling2d(inputs=convolution_3, pool_size=[2, 2], strides=2)

        with tf.variable_scope('convolution4'):
            convolution_4 = self.conv_layer(input_tensor=max_pooling_3,
                                            depth_in=256,
                                            depth_out=512,
                                            mode_name=mode_name,
                                            histogram_summary=histogram_summary,
                                            kernel_image_summary=False,
                                            activation_image_summary=activation_image_summary)

        with tf.variable_scope('max_pooling4'):
            max_pooling_4 = tf.layers.max_pooling2d(inputs=convolution_4, pool_size=[2, 2], strides=2)

        with tf.variable_scope('dense1'):
            max_pooling_3 = tf.reshape(max_pooling_4, [max_pooling_4.shape[0].value, max_pooling_4.shape[1].value * max_pooling_4.shape[2].value * max_pooling_4.shape[3].value])
            dense_1 = tf.layers.dense(inputs=max_pooling_3, units=1024, activation=activation.lrelu)

        if mode_name == self.config.MODE.TRAINING:
            tf.summary.histogram('dense1'.format(mode_name), dense_1)

        with tf.variable_scope('logits'):
            logits = tf.layers.dense(inputs=dense_1, units=self.config.NUM_CLASSES, activation=activation.lrelu)

        if mode_name == self.config.MODE.TRAINING:
            tf.summary.histogram('logits', logits)
            tf.summary.histogram('softmax', tf.nn.softmax(logits=logits))

        return logits

    def _conv2d(self, x, weights, strides):
        return tf.nn.conv2d(x, weights, strides=strides, padding='SAME')

    def conv_layer(self, mode_name, input_tensor, depth_in, depth_out, kernel_height=3, kernel_width=3, strides=(1, 1, 1, 1),
                   activation_fn=activation.lrelu, histogram_summary=False, kernel_image_summary=False, activation_image_summary=False):

        weights = tf.get_variable("weights", [kernel_height, kernel_width, depth_in, depth_out], initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable("biases", [depth_out], initializer=tf.constant_initializer(0.01))
        convolutions = self._conv2d(input_tensor, weights, strides=strides)
        activations = activation_fn(convolutions + biases, self.config.LEAKY_RELU_ALPHA)

        if histogram_summary:
            tf.summary.histogram(mode_name + '_weights', weights)
            tf.summary.histogram(mode_name + '_activations', activations)

        if kernel_image_summary:
            weights_image_grid = self.visualization.kernels_image_grid(kernel=weights)
            tf.summary.image(mode_name + '/features', weights_image_grid, max_outputs=1)

        if activation_image_summary:
            activation_image = self.visualization.activation_image(activations=activations)
            tf.summary.image("/activated", activation_image)

        return activations
