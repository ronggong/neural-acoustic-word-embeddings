from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
import logging
from os import path
from config import Config
import numpy as np
import tensorflow as tf
from model import Model
from data import Data_preprocessing
from data import Dataset
from average_precision import Ap_score


def main(margin_input, folder_model):

    config = Config(margin_input=margin_input, folder_model=folder_model)

    # initialize the logger
    logger = logging.getLogger('loss_logger')
    logger.setLevel(logging.INFO)
    hdlr = logging.FileHandler(path.join(config.logdir, 'loss.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    DATA_PREPROCESSING = Data_preprocessing()
    DATA_PREPROCESSING.process_val(config=config)

    AP_SCORE = Ap_score()

    val_data = Dataset(data=DATA_PREPROCESSING.list_feature_flatten,
                         labels=DATA_PREPROCESSING.label_integer,
                         partition="test",
                         config=config)

    # train_model = Model(is_train="train", config=config, reuse=False)
    val_model = Model(is_train="test", config=config, reuse=tf.AUTO_REUSE)

    # batch_size = config.batch_size
    batch_size = 1

    saver = tf.train.Saver()

    proto = tf.ConfigProto(intra_op_parallelism_threads=0)
    with tf.Session(config=proto) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(config.ckptdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restored from %s" % ckpt.model_checkpoint_path)

        # use val set for showing the average precision of each epoch
        embeddings, labels = [], []
        for x, ts, ids in val_data.batch(batch_size):
            embeddings.append(val_model.get_embeddings(sess, x, ts))
            labels.append(ids)
        embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
        ap = AP_SCORE.eval_pronunciation(embeddings=embeddings,
                                         labels=labels,
                                         index_student=DATA_PREPROCESSING.index_student)
        print(ap)
        # print("ap: %.4f" % average_precision(embeddings, labels))

if __name__ == "__main__":
    # margin = float(sys.argv[1])
    # folder_model = sys.argv[2]

    margin = 0.3
    folder_model = 'model_gpu'
    main(margin_input=margin, folder_model=folder_model)
