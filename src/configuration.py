class Configuration(object):
    def __init__(self):

        """
        inputs params
        """

        self.PATH = "/home/alex/Documents/Project/labeled_data"
        self.DATA_SET_PATH = "/home/alex/PycharmProjects/IndoorLocalizationSceneClassifier/data"
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

        self.VALIDATION_SIZE = 0
        self.TESTING_SIZE = 0
        self.TRAINING_SIZE = 0

        class Size(object):
            def __init__(self, width, height, channels):
                self.WIDTH = width
                self.HEIGHT = height
                self.CHANNELS = channels
                pass

        self.IMAGE_SIZE = Size(268, 32, 3)

        """
        output params
        """
        self.OUTPUT_PATH = '/home/alex/PycharmProjects/IndoorLocalizationSceneClassifier/runs/out_'
        # self.OUTPUT_PATH = '/media/alex/My Passport/0_alex_runs/out_'

        """
        modes
        """
        class Mode(object):
            def __init__(self):
                self.TRAINING = 'training'
                self.VALIDATION = 'validation'
                self.TESTING = 'testing'
                pass

        self.MODE = Mode()

        """
        training params
        """

        self.BATCH_SIZE = 256
        self.EPOCHS = 100
        self.STEPS = 10000
        self.LOG_PERIOD = 10  # steps
        self.SAVE_PERIOD = 1000  # seconds
        self.MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4
        self.NUM_PREPROCESSING_THREADS = 16
        self.NUM_EPOCHS_PER_DECAY = 10  # Epochs after which learning rate decays.
        self.LEARNING_RATE_DECAY_FACTOR = 0.005  # 0.005  # Learning rate decay factor.
        self.INITIAL_LEARNING_RATE = 0.001  # 0.001 # Initial learning rate.
        self.END_LEARNING_RATE = 0.00001  # for polynomial lr decay only

        """
        evaluation params
        """

        self.VALIDATION_PERIOD = 100  # steps
        self.TESTING_PERIOD = 500  # steps
