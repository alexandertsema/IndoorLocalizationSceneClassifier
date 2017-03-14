import os
import numpy as np
import sys
from scipy import misc


class Inputs:
    def __init__(self, config):
        self.config = config
        self.data_size = 0
        pass

    def read_files(self):
        prob = ['', '', '', '', '', '', '', '', '', '']  # TODO implement something smart to split data into sets
        for i in range(int(self.config.TESTING_PERC * 10)):
            prob[i] = 'testing_set'
        for i in range(int(self.config.TESTING_PERC * 10), int(self.config.VALIDATION_PERC * 10)):
            prob[i] = 'validation_set'
        for i in range(int(self.config.VALIDATION_PERC * 10), 10):
            prob[i] = 'training_set'

        data_set = {
            'training_set':   {'x': [],
                               'y': [],
                               'size': 0},
            'validation_set': {'x': [],
                               'y': [],
                               'size': 0},
            'testing_set':    {'x': [],
                               'y': [],
                               'size': 0}
        }

        print('Reading data from', self.config.PATH)

        for catalog in os.listdir(self.config.PATH):
            for _class in os.listdir('{0}/{1}'.format(self.config.PATH, catalog)):
                label = self._class_label(_class)
                for sample in os.listdir('{0}/{1}/{2}'.format(self.config.PATH, catalog, _class)):
                    set_name = prob[np.random.randint(0, 10)]

                    data_set[set_name]['x'].append(self.raw_bytes('{0}/{1}/{2}/{3}'.format(self.config.PATH, catalog, _class, sample)))
                    data_set[set_name]['y'].append(label)
                    data_set[set_name]['size'] += 1

                    sys.stdout.write('\r>> Samples read: {}'.format(self.data_size))
                    sys.stdout.flush()

                    self.data_size += 1

        print()
        print('Data set is {} samples'.format(self.data_size + 1))
        print('Training set is {} samples'.format(data_set['training_set']['size']))
        print('Validation set is {} samples'.format(data_set['validation_set']['size']))
        print('Testing set is {} samples'.format(data_set['testing_set']['size']))

        return data_set['training_set'], data_set['validation_set'], data_set['testing_set']

    def _class_label(self, category):
        return self.config.CLASSES[category]

    def raw_bytes(self, file_name):
        raw_image = self._decoded_image(file_name)
        return self.preprocess_image(raw_image)

    @staticmethod
    def _decoded_image(file_name):
        return misc.imread(file_name)

    def preprocess_image(self, raw_image):
        return misc.imresize(raw_image, (self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.CHANNELS), interp='bilinear', mode=None)
