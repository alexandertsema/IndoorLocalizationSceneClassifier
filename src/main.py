from datetime import datetime
import time
from configuration import Configuration
from data_set import DataSet
import tensorflow as tf
from evaluation import Evaluation
from log import Logger
from model import Model
from training import Training

session_name = input("Session name: ")

config = Configuration()  # general settings
data_sets = DataSet(config)  # data sets retrieval
model = Model(config)  # model builder
trainer = Training(config)  # training ops
evaluation = Evaluation(config)  # evaluation ops
logger = Logger(config)

with tf.Graph().as_default():
    training_set_x, training_set_y, training_size, \
    validation_set_x, validation_set_y, validation_size, \
    testing_set_x, testing_set_y, testing_size = data_sets.get_data_sets()

    #   training
    predictions_training = model.inference_(x=training_set_x, mode_name=config.MODE.TRAINING)
    loss_training = evaluation.loss(predictions=predictions_training, labels=training_set_y, mode_name=config.MODE.TRAINING)
    accuracy_training = evaluation.accuracy(predictions=predictions_training, labels=training_set_y, mode_name=config.MODE.TRAINING)
    train_op = trainer.train(loss=loss_training, global_step=tf.contrib.framework.get_or_create_global_step(), num_examples_per_epoch_for_train=1700)  # TODO get data set size

    tf.get_variable_scope().reuse_variables()

    #   validation
    predictions_validation = model.inference_(x=validation_set_x, mode_name=config.MODE.VALIDATION)
    loss_validation = evaluation.loss(predictions=predictions_validation, labels=validation_set_y, mode_name=config.MODE.VALIDATION)
    accuracy_validation = evaluation.accuracy(predictions=predictions_validation, labels=validation_set_y, mode_name=config.MODE.VALIDATION)

    # tf.get_variable_scope().reuse_variables()

    #   testing
    predictions_testing = model.inference_(x=testing_set_x, mode_name=config.MODE.TESTING)
    loss_testing = evaluation.loss(predictions=predictions_testing, labels=testing_set_y, mode_name=config.MODE.TESTING)
    accuracy_testing = evaluation.accuracy(predictions=predictions_testing, labels=testing_set_y, mode_name=config.MODE.TESTING)

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=config.OUTPUT_PATH + session_name,
            hooks=[tf.train.StopAtStepHook(last_step=config.STEPS),
                   tf.train.NanTensorHook(loss_training),
                   tf.train.NanTensorHook(accuracy_training)],
            save_checkpoint_secs=config.SAVE_PERIOD,
            save_summaries_steps=config.LOG_PERIOD,
            config=tf.ConfigProto(
                log_device_placement=False)) as mon_sess:
        step = 0
        while not mon_sess.should_stop():
            start_time = time.time()
            _, loss_training_value, accuracy_training_value = mon_sess.run([train_op, loss_training, accuracy_training])
            duration = time.time() - start_time
            logger.log(step=step, duration=duration, loss=loss_training_value, accuracy=accuracy_training_value, mode=config.MODE.TRAINING)

            if step != 0 and step % config.VALIDATION_PERIOD == 0:  # validate model and write to console
                loss_validation_value, accuracy_validation_value = mon_sess.run([loss_validation, accuracy_validation])
                logger.log(step=step, duration=duration, loss=loss_validation_value, accuracy=accuracy_validation_value, mode=config.MODE.VALIDATION)

            if step != 0 and step % config.TESTING_PERIOD == 0:  # test model and write to console
                loss_testing_value, accuracy_testing_value = mon_sess.run([loss_testing, accuracy_testing])
                logger.log(step=step, duration=duration, loss=loss_testing_value, accuracy=accuracy_testing_value, mode=config.MODE.TESTING)

            step += 1

