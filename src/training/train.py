import math
import os
import time
from datetime import datetime

import tensorflow as tf
from helpers.log import Logger
from tensorflow.python.framework.errors_impl import OutOfRangeError

from data.data_set import DataSet
from evaluation.evaluation import Evaluation
from helpers.configuration import Configuration
from helpers.sessions import Sessions
from model.cnn import Cnn
from training.training import Training


def train(session_name=None):
    if session_name is None:
        session_name = input("Session name: ")

    config = Configuration()  # general settings
    data_sets = DataSet(config)  # data sets retrieval
    model = Cnn(config)  # model builder
    trainer = Training(config)  # training ops
    evaluation = Evaluation(config)  # evaluation ops
    logger = Logger(config)

    with tf.Graph().as_default():
        data_set = data_sets.get_data_sets(config.BATCH_SIZE)

        #   training
        print('Building model...')
        predictions_training = model.inference(x=data_set.training_set.x, mode_name=config.MODE.TRAINING)
        loss_training = evaluation.loss(predictions=predictions_training, labels=data_set.training_set.y, mode_name=config.MODE.TRAINING)
        accuracy_training = evaluation.accuracy(predictions=predictions_training, labels=data_set.training_set.y, mode_name=config.MODE.TRAINING)
        global_step_tensor = tf.contrib.framework.get_or_create_global_step()
        train_op = trainer.train(loss=loss_training, global_step=global_step_tensor, num_examples_per_epoch_for_train=data_set.training_set.size)

        tf.get_variable_scope().reuse_variables()

        #   validation
        predictions_validation = model.inference(x=data_set.validation_set.x, mode_name=config.MODE.VALIDATION)
        loss_validation = evaluation.loss(predictions=predictions_validation, labels=data_set.validation_set.y, mode_name=config.MODE.VALIDATION)
        accuracy_validation = evaluation.accuracy(predictions=predictions_validation, labels=data_set.validation_set.y, mode_name=config.MODE.VALIDATION)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        print('Starting session...')
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(config.OUTPUT_PATH + session_name, sess.graph)
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sessions_helper = Sessions(config=config, session=sess, saver=saver, session_name=session_name, summary_writer=summary_writer, coordinator=coord, threads=threads)
            sessions_helper.restore()

            global_step = 0
            epoch = 0
            step = 0
            start_time = datetime.now()
            print('Starting training with {} epochs {} steps each...'.format(config.EPOCHS, int(data_set.training_set.size / config.BATCH_SIZE)))
            print()
            try:
                for epoch in range(config.EPOCHS):
                    for step in range(int(data_set.training_set.size / config.BATCH_SIZE)):
                        start_time_op = time.time()
                        _, summary, loss_training_value, accuracy_training_value = sess.run([train_op, merged, loss_training, accuracy_training])
                        duration = time.time() - start_time_op
                        global_step = tf.train.global_step(sess, global_step_tensor)
                        logger.log(global_step=global_step, epoch=epoch+1, step=step+1, duration=duration, loss=loss_training_value, accuracy=accuracy_training_value, mode=config.MODE.TRAINING)

                        if global_step % config.LOG_PERIOD == 0:  # update tensorboard
                            summary_writer.add_summary(summary, global_step)

                        if global_step == 1 or global_step % config.SAVE_PERIOD == 0:  # save model
                            sessions_helper.save(global_step_tensor=global_step_tensor, message='Initial saving...')

                        if math.isnan(loss_training_value):
                            print("loss is NaN, breaking training...")
                            exit(-1)

                        if loss_training_value <= config.TARGET_LOSS:  # early stop with good results

                            sessions_helper.save(global_step_tensor=global_step_tensor, message='Model reached {} witch is less than target loss, saving model...'.format(loss_training_value))

                            sessions_helper.end()

                            return session_name

                    # validate
                    loss_validation_value, accuracy_validation_value = sess.run([loss_validation, accuracy_validation])
                    logger.log(global_step=global_step, epoch=epoch+1, step=step+1, duration=1, loss=loss_validation_value, accuracy=accuracy_validation_value, mode=config.MODE.VALIDATION)

                sessions_helper.save(global_step_tensor=global_step_tensor, message='OutOfRangeError occurred, saving model...')
                print("Restarting training...")
                train(session_name)

            except KeyboardInterrupt:

                sessions_helper.save(global_step_tensor=global_step_tensor, message='User requested to stop training, saving model...')
                sessions_helper.end()

                return session_name

            sessions_helper.save(global_step_tensor=global_step_tensor, message="Training finished in {}, saving model...".format(datetime.now() - start_time))
            sessions_helper.end()

            return session_name





