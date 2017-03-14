from datetime import datetime
import time
from configuration import Configuration
from data_set import DataSet
import tensorflow as tf
from evaluation import Evaluation
from model import Model
from training import Training

config = Configuration()  # general settings
data_sets = DataSet(config)  # data sets retrieval
model = Model(config)  # model builder
trainer = Training(config)  # training ops
evaluation = Evaluation()   # evaluation ops

with tf.Graph().as_default():
    training_set_x, training_set_y, validation_set_x, validation_set_y, testing_set_x, testing_set_y = data_sets.get_data_sets()
    prediction = model.inference(x=training_set_x)
    loss = trainer.loss(logits=prediction, labels=training_set_y)
    accuracy = trainer.accuracy(logits=prediction, labels=training_set_y)
    train_op = trainer.train(loss=loss, global_step=tf.contrib.framework.get_or_create_global_step(), NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=17000)  # TODO get real number of examples

    # saver = tf.train.Saver()
    # sess = tf.Session()
    #
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #
    # sess.run(init_op)
    #
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    # print(sess.run([accuracy]))  # Here goes training and eval
    # # sess.run(train_op)
    #
    # coord.join(threads)
    # sess.close()

    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""

        def begin(self):
            self._step = -1

        def before_run(self, run_context):
            self._step += 1
            self._start_time = time.time()
            return tf.train.SessionRunArgs({'loss': loss, 'accuracy': accuracy})  # Asks for loss accuracy value.

        def after_run(self, run_context, run_values):
            duration = time.time() - self._start_time
            loss_value = run_values.results['loss']
            accuracy_value = run_values.results['accuracy']
            if self._step % config.LOG_PERIOD == 0:
                num_examples_per_step = config.BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f, accuracy = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), self._step, loss_value, accuracy_value,
                                    examples_per_sec, sec_per_batch))


    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=config.OUTPUT_PATH,
            hooks=[tf.train.StopAtStepHook(last_step=config.STEPS),
                   tf.train.NanTensorHook(loss),
                   tf.train.NanTensorHook(accuracy),
                   _LoggerHook()],
            save_checkpoint_secs=config.SAVE_PERIOD,
            save_summaries_steps=config.LOG_PERIOD,
            config=tf.ConfigProto(
                log_device_placement=False)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)

            # print(mon_sess.run(prediction))
            # print(mon_sess.run(training_set_y))

