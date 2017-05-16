class DataSet(object):
    def __init__(self):
        class TrainingSet(object):
            def __init__(self):
                self.x = None
                self.y = None
                self.size = 0

        class ValidationSet(object):
            def __init__(self):
                self.x = None
                self.y = None
                self.size = 0

        class TestingSet(object):
            def __init__(self):
                self.x = None
                self.y = None
                self.size = 0

        self.training_set = TrainingSet()
        self.validation_set = ValidationSet()
        self.testing_set = TestingSet()
        pass
