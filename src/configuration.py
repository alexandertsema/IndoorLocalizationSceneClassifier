class Configuration(object):
    def __init__(self):

        """
        inputs params
        """

        self.PATH = "/home/alex/Documents/Project/SceneClassifier/data"
        self.DATA_SET_PATH = "/home/alex/PycharmProjects/Project/data"
        self.CLASSES = {
            'Bathroom': 0,
            'Doors': 1,
            'Exit': 2,
            'ShortCorridor': 3,
            'CorridorWithBoards': 4,
            'Elevator': 5,
            'LongCorridor': 6,
            'Windows': 7
        }
        self.NUM_CLASSES = self.CLASSES.__len__()
        self.VALIDATION_PERC = 0.2
        self.TESTING_PERC = 0.1
        self.TRAINING_PERC = 1 - self.VALIDATION_PERC - self.TESTING_PERC

        class Size(object):
            def __init__(self, width, height, channels):
                self.WIDTH = width
                self.HEIGHT = height
                self.CHANNELS = channels
            pass

        self.IMAGE_SIZE = Size(268, 32, 3)

        """
        training params
        """

        self.BATCH_SIZE = 128
        self.EPOCHS = 1000
        self.MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4
        self.SHUFFLE_BATCH = True
        self.NUM_PREPROCESSING_THREADS = 16
        self.MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
        self.NUM_EPOCHS_PER_DECAY = 100  # Epochs after which learning rate decays.
        self.LEARNING_RATE_DECAY_FACTOR = 0.0005  # Learning rate decay factor.
        self.INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.
