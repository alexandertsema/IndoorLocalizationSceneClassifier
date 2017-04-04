import sys
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from data.data_set import DataSet
from evaluation.evaluation import Evaluation
from helpers.configuration import Configuration
from helpers.sessions import Sessions
from model.cnn import Cnn
from visualization.visualization import Visualization


def test(session_name=None, is_visualize=False):
    if session_name is None:
        session_name = input("Session name: ")

    config = Configuration()  # general settings
    data_sets = DataSet(config)  # data sets retrieval
    model = Cnn(config)  # model builder
    evaluation = Evaluation(config)
    visualization = Visualization(config)

    with tf.Graph().as_default():
        data_set = data_sets.get_data_sets(config.TESTING_BATCH_SIZE)

        print('Building model...')
        predictions_testing = model.inference(x=data_set.testing_set.x, mode_name=config.MODE.TESTING)
        is_correct = evaluation.correct_number(predictions_testing, data_set.testing_set.y)
        predictions_testing = tf.argmax(predictions_testing, 1)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        print('Starting session...')
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(config.OUTPUT_PATH + session_name + '_tested', sess.graph)
            sessions_helper = Sessions(config=config, session=sess, saver=saver, session_name=session_name, summary_writer=summary_writer, coordinator=coord, threads=threads)

            sessions_helper.restore()

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

                    example_image, actual_label, predicted_label, is_correct_result = sess.run([data_set.testing_set.x, data_set.testing_set.y, predictions_testing, is_correct])
                    true_count += np.sum(is_correct_result)

                    labels.append(actual_label)
                    predictions.append(predicted_label)

                    if is_visualize:
                        visualization.show_example(predicted_label, actual_label, example_image, is_correct_result)

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

            sessions_helper.end()

