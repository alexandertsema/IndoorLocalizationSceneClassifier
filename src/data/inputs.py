import os
import random
import sys

from scipy import misc

from data.abstract.data_set import DataSet


class Inputs(DataSet):
    def __init__(self, config):
        DataSet.__init__(self)
        self.config = config
        self.training_set.x = []
        self.validation_set.x = []
        self.testing_set.x = []
        self.training_set.y = []
        self.validation_set.y = []
        self.testing_set.y = []
        pass

    def read_files(self):
        print('Reading data from', self.config.PATH)
        delta = abs(self.config.VALIDATION_PERC - self.config.TESTING_PERC)
        val_perc = 0.5 + delta
        for catalog in os.listdir(self.config.PATH):
            for _class in os.listdir('{0}/{1}'.format(self.config.PATH, catalog)):
                label = self._class_label(_class)
                for sample in os.listdir('{0}/{1}/{2}'.format(self.config.PATH, catalog, _class)):
                    if self.biased_random(self.config.TRAINING_PERC):
                        data_set = self.training_set
                    elif self.biased_random(val_perc):
                        data_set = self.validation_set
                    else:
                        data_set = self.testing_set

                    data_set.x.append(self.raw_bytes('{0}/{1}/{2}/{3}'.format(self.config.PATH, catalog, _class, sample)))
                    data_set.y.append(label)
                    data_set.size += 1

                    sys.stdout.write('\r>> Samples read: {}'.format(self.training_set.size + self.validation_set.size + self.testing_set.size))
                    sys.stdout.flush()

        print()
        print('Data set is {} samples'.format(self.training_set.size + self.validation_set.size + self.testing_set.size))
        print('Training set is {} samples'.format(self.training_set.size))
        print('Validation set is {} samples'.format(self.validation_set.size))
        print('Testing set is {} samples'.format(self.testing_set.size))

        return self.training_set, self.validation_set, self.testing_set

    def _class_label(self, category):
        return self.config.CLASSES[category]

    @staticmethod
    def biased_random(prob_true=0.5):
        return random.random() < prob_true

    def raw_bytes(self, file_name):
        raw_image = self._decoded_image(file_name)
        return self.preprocess_image(raw_image)

    @staticmethod
    def _decoded_image(file_name):
        return misc.imread(file_name)

    def preprocess_image(self, raw_image):
        #  TODO could add distortions here
        return misc.imresize(raw_image, (self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.CHANNELS), interp='bilinear', mode=None)
