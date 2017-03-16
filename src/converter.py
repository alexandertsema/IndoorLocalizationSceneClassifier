import os
import numpy as np
import tensorflow as tf
import sys


class Converter:

    def __init__(self, config):
        self.config = config
        pass

    def convert_to_tf_records(self, data_set, name):
        images = np.array(data_set['x'])
        labels = np.array(data_set['y'])
        num_examples = data_set['size']

        if images.shape[0] != num_examples:
            raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], num_examples))
        rows = images.shape[1]
        cols = images.shape[2]
        depth = images.shape[3]

        filename = os.path.join(self.config.DATA_SET_PATH, name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            sys.stdout.write('\r>> Samples converted: {}'.format(index))
            sys.stdout.flush()
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height':    self._int64_feature(rows),
                'width':     self._int64_feature(cols),
                'depth':     self._int64_feature(depth),
                'label':     self._int64_feature(int(labels[index])),
                #'size':      self._int64_feature(num_examples),
                'image_raw': self._bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()
        print()
        pass

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))