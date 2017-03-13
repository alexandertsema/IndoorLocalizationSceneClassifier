from configuration import Configuration
from data_set import DataSet
import tensorflow as tf


config = Configuration()  # general settings
data_sets = DataSet(config)  # data sets retrieval

with tf.Graph().as_default():
    training_set_x, training_set_y, validation_set_x, validation_set_y, testing_set_x, testing_set_y = data_sets.get_data_sets()

    sess = tf.Session()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print(sess.run([testing_set_x]))  # Here goes training and eval4

    coord.join(threads)
    sess.close()

