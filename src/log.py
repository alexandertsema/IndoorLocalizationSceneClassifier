from datetime import datetime


class Logger:
    def __init__(self, config):
        self.config = config
        pass

    def log(self, step, duration, loss, accuracy, mode):
        if mode == self.config.MODE.TRAINING:
            num_examples_per_step = self.config.BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = '%s: step %d, loss = %.4f, accuracy = %.4f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (datetime.now(), step, loss, accuracy, examples_per_sec, sec_per_batch))

        if mode == self.config.MODE.VALIDATION:
            print()
            format_str = '>> VALIDATION: %s: step %d, loss = %.4f, accuracy = %.4f'
            print(format_str % (datetime.now(), step, loss, accuracy))
            print()

        if mode == self.config.MODE.TESTING:
            print()
            format_str = '>> TESTING: %s: step %d, loss = %.4f, accuracy = %.4f'
            print(format_str % (datetime.now(), step, loss, accuracy))
            print()
        pass
