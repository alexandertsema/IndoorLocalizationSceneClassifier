import tensorflow as tf
from tensorlayer import activation
from math import *
import matplotlib.pyplot as plt


class Model:
    def __init__(self, config):
        self.config = config
        self.initial_weight_deviation = 0.01
        self.leaky_relu_leakiness = 0.1
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

    def inference_(self, x, mode_name):
        with tf.name_scope('inputs'):
            x = tf.reshape(x, [-1, self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.CHANNELS])

        tf.summary.image("/inputs", x, max_outputs=4)

        with tf.variable_scope('convolution1'):
            convolution_1 = self.conv_layer(input_tensor=x,
                                            depth_in=self.config.IMAGE_SIZE.CHANNELS,
                                            depth_out=96,
                                            kernel_height=9, kernel_width=9,
                                            mode_name=mode_name,
                                            histogram_summary=True,
                                            kernel_image_summary=True,
                                            activation_image_summary=True)

        with tf.variable_scope('max_pooling1'):
            max_pooling_1 = tf.layers.max_pooling2d(inputs=convolution_1, pool_size=[2, 2], strides=2)

        with tf.variable_scope('convolution2'):
            convolution_2 = self.conv_layer(input_tensor=max_pooling_1,
                                            depth_in=96,
                                            depth_out=256,
                                            kernel_height=5, kernel_width=5,
                                            mode_name=mode_name,
                                            histogram_summary=True,
                                            kernel_image_summary=False,
                                            activation_image_summary=True)

        with tf.variable_scope('max_pooling2'):
            max_pooling_2 = tf.layers.max_pooling2d(inputs=convolution_2, pool_size=[2, 2], strides=2)

        with tf.variable_scope('convolution3'):
            convolution_3 = self.conv_layer(input_tensor=max_pooling_2,
                                            depth_in=256,
                                            depth_out=384,
                                            mode_name=mode_name,
                                            histogram_summary=True,
                                            kernel_image_summary=False,
                                            activation_image_summary=True)

        with tf.variable_scope('convolution4'):
            convolution_4 = self.conv_layer(input_tensor=convolution_3,
                                            depth_in=384,
                                            depth_out=384,
                                            mode_name=mode_name,
                                            histogram_summary=True,
                                            kernel_image_summary=False,
                                            activation_image_summary=True)

        with tf.variable_scope('convolution5'):
            convolution_5 = self.conv_layer(input_tensor=convolution_4,
                                            depth_in=384,
                                            depth_out=256,
                                            mode_name=mode_name,
                                            histogram_summary=True,
                                            kernel_image_summary=False,
                                            activation_image_summary=True)

        with tf.variable_scope('dense1'):
            convolution_5_reshaped = tf.reshape(convolution_5, [convolution_5.shape[0].value, convolution_5.shape[1].value * convolution_5.shape[2].value * convolution_5.shape[3].value])
            dense_1 = tf.layers.dense(inputs=convolution_5_reshaped, units=512, activation=activation.lrelu)

        with tf.variable_scope('dense2'):
            dense_2 = tf.layers.dense(inputs=dense_1, units=512, activation=activation.lrelu)

        with tf.variable_scope('logits'):
            logits = tf.layers.dense(inputs=dense_2, units=self.config.NUM_CLASSES, activation=activation.lrelu)

        tf.summary.histogram('rawlogits_{}'.format(mode_name), logits)
        tf.summary.histogram('classesprobdistributionprediction_{}'.format(mode_name), tf.nn.softmax(logits=logits))

        return logits

    def _conv2d(self, x, weights, strides):
        return tf.nn.conv2d(x, weights, strides=strides, padding='SAME')

    def conv_layer(self, mode_name, input_tensor, depth_in, depth_out, kernel_height=3, kernel_width=3, strides=(1, 1, 1, 1),
                   activation_fn=activation.lrelu, histogram_summary=False, kernel_image_summary=False, activation_image_summary=False):

        weights = tf.get_variable("weights", [kernel_height, kernel_width, depth_in, depth_out], initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable("biases", [depth_out], initializer=tf.constant_initializer(0.01))
        convolutions = self._conv2d(input_tensor, weights, strides=strides)
        activations = activation_fn(convolutions + biases, self.leaky_relu_leakiness)

        if histogram_summary:
            tf.summary.histogram(mode_name + '_weights', weights)
            tf.summary.histogram(mode_name + '_activations', activations)

        if kernel_image_summary:
            weights_image_grid = self.put_kernels_on_grid(kernel=weights)
            tf.summary.image(mode_name + '/features', weights_image_grid, max_outputs=1)

        if activation_image_summary:
            activation_image = self.activation_image(activations=activations)
            tf.summary.image("/activated", activation_image)

        return activations

    @staticmethod
    def put_kernels_on_grid(kernel, pad=1):  # 1st layer only
        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1:
                        print('Who would enter a prime number of filters')
                    return (i, int(n / i))

        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
        # print('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)

        kernel1 = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel1.get_shape()[0] + 2 * pad
        X = kernel1.get_shape()[1] + 2 * pad

        channels = kernel1.get_shape()[2]

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scaling to [0, 255] is not necessary for tensorboard
        return x7

    @staticmethod
    def activation_image(activations):
        layer1_image1 = activations[0:3, :, :, 0:4]  # 4 - number images to show, 3 - number channels
        layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
        padding = 4
        layer1_image1 = tf.pad(layer1_image1, tf.constant([[0, 0], [int(padding/2), int(padding/2)], [padding, padding], [0, 0]]), mode='CONSTANT')
        list_lc1 = tf.split(axis=0, num_or_size_splits=4, value=layer1_image1)

        layer_combine_1 = tf.concat(axis=1, values=list_lc1)

        return layer_combine_1
