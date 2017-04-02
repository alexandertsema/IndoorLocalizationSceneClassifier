import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.data_set import DataSet
from helpers.configuration import Configuration
from model.cnn import Cnn


def test(session_name=None):
    if session_name is None:
        session_name = input("Session name: ")

    config = Configuration()  # general settings
    data_sets = DataSet(config)  # data sets retrieval
    model = Cnn(config)  # model builder

    with tf.Graph().as_default():
        data_set = data_sets.get_data_sets(config.TESTING_BATCH_SIZE)

        print('Building model...')
        predictions_testing = model.inference(x=data_set.testing_set.x, mode_name=config.MODE.TESTING)
        top_k_op = tf.nn.in_top_k(predictions_testing, data_set.testing_set.y, 1)
        prediction = tf.argmax(predictions_testing, 1)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        print('Starting session...')
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if os.path.exists(config.OUTPUT_PATH + session_name):
                checkpoint = tf.train.get_checkpoint_state(config.OUTPUT_PATH + session_name)
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Model restored from: %s" % checkpoint.model_checkpoint_path)
            summary_writer = tf.summary.FileWriter(config.OUTPUT_PATH + session_name + '_tested', sess.graph)
            print()
            true_count = 0
            summary = None
            start_time = time.time()
            labels = []
            predictions = []
            for epoch in range(config.TESTING_EPOCHS):
                for step in range(int(data_set.testing_set.size / config.TESTING_BATCH_SIZE)):
                    summary = sess.run(merged)

                    sys.stdout.write('\r>> Examples tested: {}/{}'.format(step, int(data_set.testing_set.size / config.TESTING_BATCH_SIZE)))
                    sys.stdout.flush()

                    actual_label, predicted_label, actual_vector, predicted_vector, is_correct = sess.run([data_set.testing_set.y, prediction, tf.one_hot(data_set.testing_set.y, config.NUM_CLASSES),
                                                                                                           predictions_testing, top_k_op])
                    true_count += np.sum(is_correct)

                    labels.append(actual_label)
                    predictions.append(predicted_label)

            summary_writer.add_summary(summary, 1)

            np_labels = np.array(labels)
            np_predictions = np.array(predictions)

            conf_matrix = tf.confusion_matrix(labels=tf.squeeze(np_labels), predictions=tf.squeeze(np_predictions), num_classes=config.NUM_CLASSES)
            print()
            c_m = sess.run(conf_matrix)
            print(c_m)

            precision = true_count / data_set.testing_set.size
            print()
            print('testing completed in %s' % (time.time() - start_time))
            print('%s: accuracy @ 1 = %.3f' % (datetime.now(), precision * 100))

            print("testing finished, releasing resources...")
            summary_writer.close()
            print("summary writer closed")
            sess.close()
            print("session closed")
            print("resources released...")
            print("end")
            exit()


def test_visualized(session_name):
    if session_name is None:
        session_name = input("Session name: ")

    config = Configuration()  # general settings
    data_sets = DataSet(config)  # data sets retrieval
    model = Model(config)  # model builder

    with tf.Graph().as_default():
        data_set = data_sets.get_data_sets(config.TESTING_BATCH_SIZE)

        print('Building model...')

        predictions_testing = model.inference(x=data_set.testing_set.x, mode_name=config.MODE.TESTING)

        top_k_op = tf.nn.in_top_k(predictions_testing, data_set.testing_set.y, 1)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        print('Starting session...')
        with tf.Session() as sess:

            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            if os.path.exists(config.OUTPUT_PATH + session_name):
                checkpoint = tf.train.get_checkpoint_state(config.OUTPUT_PATH + session_name)
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Model restored from: %s" % checkpoint.model_checkpoint_path)

            summary_writer = tf.summary.FileWriter(config.OUTPUT_PATH + session_name + '_tested', sess.graph)

            print()
            true_count = 0
            summary = None
            start_time = time.time()
            for epoch in range(config.TESTING_EPOCHS):
                for step in range(int(data_set.testing_set.size / config.TESTING_BATCH_SIZE)):

                    img = sess.run(data_set.testing_set.x)
                    example_category = None
                    label = sess.run(data_set.testing_set.y)[0]
                    for k, v in config.CLASSES.items():
                        if v == label:
                            example_category = k
                            break

                    img = np.reshape(img, (config.IMAGE_SIZE.HEIGHT, config.IMAGE_SIZE.WIDTH, config.IMAGE_SIZE.CHANNELS))

                    summary, predictions = sess.run([merged, top_k_op])

                    print('current example belongs to class {}: {}'.format(example_category, predictions))

                    imgplot = plt.imshow(img)
                    plt.show()

                    true_count += np.sum(predictions)

                    # sys.stdout.write('\r>> Examples tested: {}/{}'.format(step, int(data_set.testing_set.size / config.TESTING_BATCH_SIZE)))
                    # sys.stdout.flush()

            summary_writer.add_summary(summary, 1)

            precision = true_count / config.TESTING_SIZE

            print('testing completed in %s' % (time.time() - start_time))
            print('%s: accuracy @ 1 = %.3f' % (datetime.now(), precision * 100))

            print("testing finished, releasing resources...")
            summary_writer.close()
            print("summary writer closed")
            sess.close()
            print("session closed")
            print("resources released...")
            print("end")
            exit()


# test_visualized('train_4 (copy)')
#test('train_4')
