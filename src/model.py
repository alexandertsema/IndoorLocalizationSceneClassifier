import tensorflow as tf


class Model:

    def __init__(self, config):
        self.config = config
        pass

    # def inference(self, x):
    #     # conv1
    #     with tf.variable_scope('conv1') as scope:
    #         kernel = self._variable_with_weight_decay('weights',
    #                                              shape=[5, 5, 2, 64],
    #                                              stddev=5e-2,
    #                                              wd=0.0)
    #         conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    #         biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    #         pre_activation = tf.nn.bias_add(conv, biases)
    #         conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #         # _activation_summary(conv1)
    #
    #     # pool1
    #     pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
    #                            padding='SAME', name='pool1')
    #     # norm1
    #     norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                       name='norm1')
    #
    #     # conv2
    #     with tf.variable_scope('conv2') as scope:
    #         kernel = self._variable_with_weight_decay('weights',
    #                                              shape=[5, 5, 64, 64],
    #                                              stddev=5e-2,
    #                                              wd=0.0)
    #         conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    #         biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    #         pre_activation = tf.nn.bias_add(conv, biases)
    #         conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #         # _activation_summary(conv2)
    #
    #     # norm2
    #     norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                       name='norm2')
    #     # pool2
    #     pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
    #                            strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    #
    #     # local3
    #     with tf.variable_scope('local3') as scope:
    #         # Move everything into depth so we can perform a single matrix multiply.
    #         reshape = tf.reshape(pool2, [self.config.BATCH_SIZE, -1])
    #         dim = reshape.get_shape()[1].value
    #         weights = self._variable_with_weight_decay('weights', shape=[dim, 384],
    #                                               stddev=0.04, wd=0.004)
    #         biases = self._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    #         local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    #         # _activation_summary(local3)
    #
    #     # local4
    #     with tf.variable_scope('local4') as scope:
    #         weights = self._variable_with_weight_decay('weights', shape=[384, 192],
    #                                               stddev=0.04, wd=0.004)
    #         biases = self._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    #         local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    #         # _activation_summary(local4)
    #
    #     # linear layer(WX + b),
    #     # We don't apply softmax here because
    #     # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    #     # and performs the softmax internally for efficiency.
    #     with tf.variable_scope('softmax_linear') as scope:
    #         weights = self._variable_with_weight_decay('weights', [192, self.config.NUM_CLASSES],
    #                                               stddev=1 / 192.0, wd=0.0)
    #         biases = self._variable_on_cpu('biases', [self.config.NUM_CLASSES],
    #                                   tf.constant_initializer(0.0))
    #         softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    #         # _activation_summary(softmax_linear)
    #
    #     return softmax_linear



    # def _variable_with_weight_decay(self, name, shape, stddev, wd):
    #     """Helper to create an initialized Variable with weight decay.
    #     Note that the Variable is initialized with a truncated normal distribution.
    #     A weight decay is added only if one is specified.
    #     Args:
    #       name: name of the variable
    #       shape: list of ints
    #       stddev: standard deviation of a truncated Gaussian
    #       wd: add L2Loss weight decay multiplied by this float. If None, weight
    #           decay is not added for this Variable.
    #     Returns:
    #       Variable Tensor
    #     """
    #     dtype = tf.float32
    #     var = self._variable_on_cpu(
    #         name,
    #         shape,
    #         tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    #     if wd is not None:
    #         weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    #         tf.add_to_collection('losses', weight_decay)
    #     return var

    def inference(self, x):
        # Input Layer
        input_layer = tf.reshape(x, [-1, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.CHANNELS])
        with tf.name_scope('conv1'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[5, 5],
                padding='SAME',
                activation=tf.nn.relu)

        with tf.name_scope('pool1'):
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        with tf.name_scope('conv2'):
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

        with tf.name_scope('pool2'):
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1].value * pool2.shape[2].value * pool2.shape[3].value])
        with tf.name_scope('dense'):
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)  # TODO switcher for modes (TRAINING, VALIDATION, TESTING)

            # Logits Layer
        with tf.name_scope('logits'):
            logits = tf.layers.dense(inputs=dropout, units=10)

            with tf.name_scope('softmax'):
                softmax = tf.nn.softmax(logits)

        return softmax
