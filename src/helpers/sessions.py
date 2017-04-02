import os
import tensorflow as tf


class Sessions:
    def __init__(self, config, session_name, session, saver, summary_writer, coordinator, threads):
        self.config = config
        self.session_name = session_name
        self.session = session
        self.saver = saver
        self.summary_writer = summary_writer
        self.coordinator = coordinator
        self.threads = threads
        pass

    def save(self, global_step_tensor, message=''):
        print(message)
        if not os.path.exists(self.config.OUTPUT_PATH + self.session_name):
            os.makedirs(self.config.OUTPUT_PATH + self.session_name)
        model_path = self.config.OUTPUT_PATH + self.session_name + '/' + 'model.ckpt'
        self.saver.save(sess=self.session, save_path=model_path, global_step=global_step_tensor)
        print("Model saved as: %s" % model_path)
        pass

    def restore(self):
        if os.path.exists(self.config.OUTPUT_PATH + self.session_name):
            checkpoint = tf.train.get_checkpoint_state(self.config.OUTPUT_PATH + self.session_name)
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Model restored from: %s" % checkpoint.model_checkpoint_path)
        pass

    def end(self):
        print("Releasing resources...")
        self.summary_writer.close()
        print("Summary writer closed")
        self.coordinator.request_stop()
        self.coordinator.join(self.threads)
        print("Threads terminated")
        self.session.close()
        print("Session closed")
        print("Exiting...")
        pass
