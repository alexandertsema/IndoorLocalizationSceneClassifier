import os

from converter import Converter
from inputs import Inputs
import tensorflow as tf


class DataSet(object):

    def __init__(self, config):
        self.config = config
        pass

    def get_data_sets(self):
        training_set_x,  training_set_y, training_size = self.read_tf_records_file('training_set')
        validation_set_x, validation_set_y, validation_size = self.read_tf_records_file('validation_set')
        testing_set_x, testing_set_y, testing_size = self.read_tf_records_file('testing_set')

        if training_set_x is None or validation_set_x is None or testing_set_x is None\
                or training_set_y is None or validation_set_y is None or testing_set_y is None:
            print()
            print('Data sets do not exist, creating new ones...')

            inputs = Inputs(self.config)
            converter = Converter(self.config)

            training_inputs, validation_inputs, testing_inputs = inputs.read_files()

            print('Converting training_set...')
            converter.convert_to_tf_records(training_inputs, 'training_set')
            print('Converting validation_set...')
            converter.convert_to_tf_records(validation_inputs, 'validation_set')
            print('Converting testing_set...')
            converter.convert_to_tf_records(testing_inputs, 'testing_set')
            self.get_data_sets()

        # self.config.VALIDATION_SIZE = validation_size
        # self.config.TESTING_SIZE = testing_size
        # self.config.TRAINING_SIZE = training_size

        return training_set_x, training_set_y, training_size, validation_set_x, validation_set_y, validation_size, testing_set_x, testing_set_y, testing_size

    def read_tf_records_file(self, name):
        filename = os.path.join(self.config.DATA_SET_PATH, name + '.tfrecords')

        if not os.path.exists(filename):
            return None, None, -1

        with tf.name_scope('input'):
            print()
            print("Creating queue for {}...".format(name))
            filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=self.config.EPOCHS)

            # Even when reading in multiple threads, share the filename
            # queue.
            image, label = self.read_and_decode(filename_queue)

            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            print("Creating batches for {}...".format(name))
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label], batch_size=self.config.BATCH_SIZE, num_threads=2,
                capacity=1000 + 3 * self.config.BATCH_SIZE,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)

            return images, sparse_labels, -1

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label':     tf.FixedLenFeature([], tf.int64),
                #'size':      tf.FixedLenFeature([], tf.int64)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self.config.IMAGE_SIZE.WIDTH * self.config.IMAGE_SIZE.HEIGHT * self.config.IMAGE_SIZE.CHANNELS])

        # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)

        #size = tf.cast(features['size'], tf.int32)

        return image, label, #size
