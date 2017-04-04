import tensorflow as tf
from math import *
import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    def __init__(self, config):
        self.config = config
        pass

    def activation_image(self, activations, images_number=4):
        images = activations[0:self.config.IMAGE_SIZE.CHANNELS, :, :, 0:images_number]
        images = tf.transpose(images, perm=[3, 1, 2, 0])
        padding = 4
        images = tf.pad(images, tf.constant([[0, 0], [int(padding / 2), int(padding / 2)], [padding, padding], [0, 0]]), mode='CONSTANT')
        list_images = tf.split(axis=0, num_or_size_splits=4, value=images)

        activated_images = tf.concat(axis=1, values=list_images)

        return activated_images

    @staticmethod
    def kernels_image_grid(kernel, pad=1):  # 1st layer only
        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1:
                        print('Who would enter a prime number of filters')
                    return i, int(n / i)

        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)

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

        return x7

    def show_example(self, predicted_label, actual_label, example_image, is_correct_result):
        example_class_predicted = self.get_example_class(predicted_label, self.config.CLASSES)
        example_class_actual = self.get_example_class(actual_label, self.config.CLASSES)
        example_image = np.reshape(np.array(example_image), (self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.CHANNELS))

        print()
        print('Predicted as {}, actually is {}. {}'.format(example_class_predicted, example_class_actual, is_correct_result))

        plt.imshow(example_image)
        plt.show()

        pass

    @staticmethod
    def get_example_class(label, classes):
        example_category = None
        for k, v in classes.items():
            if v == label:
                example_category = k
                break
        return example_category
