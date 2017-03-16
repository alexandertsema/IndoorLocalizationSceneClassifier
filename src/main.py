from datetime import datetime
import time
from configuration import Configuration
from data_set import DataSet
import tensorflow as tf
from evaluation import Evaluation
from model import Model
from training import Training

session_name = input("Session name: ")

config = Configuration()  # general settings
data_sets = DataSet(config)  # data sets retrieval
model = Model(config)  # model builder
trainer = Training(config)  # training ops
evaluation = Evaluation(config)  # evaluation ops

with tf.Graph().as_default():
    training_set_x, training_set_y, training_size, \
    validation_set_x, validation_set_y, validation_size, \
    testing_set_x, testing_set_y, testing_size = data_sets.get_data_sets()

    #   training
    predictions_training = model.inference_(x=training_set_x, mode_name=config.MODE.TRAINING)
    loss_training = evaluation.loss(predictions=predictions_training, labels=training_set_y, mode_name=config.MODE.TRAINING)
    accuracy_training = evaluation.accuracy(predictions=predictions_training, labels=training_set_y, mode_name=config.MODE.TRAINING)
    train_op = trainer.train(loss=loss_training, global_step=tf.contrib.framework.get_or_create_global_step(), num_examples_per_epoch_for_train=1700)  # TODO get data set size

    #   validation
    predictions_validation = model.inference_(x=validation_set_x, mode_name=config.MODE.VALIDATION)
    loss_validation = evaluation.loss(predictions=predictions_validation, labels=validation_set_y, mode_name=config.MODE.VALIDATION)
    accuracy_validation = evaluation.accuracy(predictions=predictions_validation, labels=validation_set_y, mode_name=config.MODE.VALIDATION)

    #   testing
    predictions_testing = model.inference_(x=testing_set_x, mode_name=config.MODE.TESTING)
    loss_testing = evaluation.loss(predictions=predictions_testing, labels=testing_set_y, mode_name=config.MODE.TESTING)
    accuracy_testing = evaluation.accuracy(predictions=predictions_testing, labels=testing_set_y, mode_name=config.MODE.TESTING)


    class _LoggerHook(tf.train.SessionRunHook):

        def begin(self):
            self._step = -1

        def before_run(self, run_context):
            self._step += 1
            self._start_time = time.time()
            return tf.train.SessionRunArgs(  # asks for loss, accuracy value.
                {
                    'loss_training':       loss_training,
                    'accuracy_training':   accuracy_training,

                    'loss_validation':     loss_validation,
                    'accuracy_validation': accuracy_validation,

                    'loss_testing':        loss_testing,
                    'accuracy_testing':    accuracy_testing
                }
            )

        def after_run(self, run_context, run_values):
            duration = time.time() - self._start_time

            loss_training_value = run_values.results['loss_training']
            accuracy_training_value = run_values.results['accuracy_training']

            loss_validation_value = run_values.results['loss_validation']
            accuracy_validation_value = run_values.results['accuracy_validation']

            loss_testing_value = run_values.results['loss_testing']
            accuracy_testing_value = run_values.results['accuracy_testing']

            if self._step % config.LOG_PERIOD == 0:  # write TRAINING loss and acc to console
                num_examples_per_step = config.BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = '%s: step %d, loss = %.4f, accuracy = %.4f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), self._step, loss_training_value, accuracy_training_value, examples_per_sec, sec_per_batch))

            if self._step != 0 and self._step % config.VALIDATION_PERIOD == 0:  # validate model and write to console
                format_str = '>> VALIDATION: %s: step %d, loss = %.4f, accuracy = %.4f'
                print()
                print(format_str % (datetime.now(), self._step, loss_validation_value, accuracy_validation_value))
                print()

            if self._step != 0 and self._step % config.TESTING_PERIOD == 0:  # validate model and write to console
                format_str = '>> TESTING: %s: step %d, loss = %.4f, accuracy = %.4f'
                print()
                print(format_str % (datetime.now(), self._step, loss_testing_value, accuracy_testing_value))
                print()


    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=config.OUTPUT_PATH + session_name,
            hooks=[tf.train.StopAtStepHook(last_step=config.STEPS),
                   tf.train.NanTensorHook(loss_training),
                   tf.train.NanTensorHook(accuracy_training),
                   _LoggerHook()],
            save_checkpoint_secs=config.SAVE_PERIOD,
            save_summaries_steps=config.LOG_PERIOD,
            config=tf.ConfigProto(
                log_device_placement=False)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)
