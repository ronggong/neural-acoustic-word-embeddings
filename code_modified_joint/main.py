from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
import sys
import logging
from os import path
from config import Config
import numpy as np
import tensorflow as tf
from model import Model
from data import Data_preprocessing
from data import Dataset
# from average_precision import average_precision


def main(margin_input, output_shape, num_layers, mtl, folder_model):

    config = Config(margin_input=margin_input,
                    output_shape=output_shape,
                    num_layers=num_layers,
                    mtl=mtl,
                    folder_model=folder_model)

    # initialize the logger
    logger = logging.getLogger('loss_logger')
    logger.setLevel(logging.INFO)
    hdlr = logging.FileHandler(path.join(config.logdir, 'loss.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    DATA_PREPROCESSING = Data_preprocessing()
    list_feature_fold_train, labels_integer_fold_train, list_feature_fold_val, labels_integer_fold_val = \
        DATA_PREPROCESSING.process_train(config=config)

    train_data = Dataset(data=list_feature_fold_train,
                         labels=labels_integer_fold_train,
                         partition="train",
                         config=config)

    dev_data = Dataset(data=list_feature_fold_val,
                       labels=labels_integer_fold_val,
                       partition="dev",  # we set it to train because we need the full output: data, lens, ...
                       config=config)

    train_model = Model(is_train="train", config=config, reuse=None)
    dev_model = Model(is_train="dev", config=config, reuse=True)
    # test_model = Model(is_train="test", config=config, reuse=True)

    batch_size = config.batch_size

    saver = tf.train.Saver()

    proto = tf.ConfigProto(intra_op_parallelism_threads=0)
    with tf.Session(config=proto) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(config.ckptdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restored from %s" % ckpt.model_checkpoint_path)

        best_loss = 100.0
        stopping_step = 0
        for epoch in range(config.current_epoch, config.num_epochs):
            print("epoch: ", epoch)

            batch = 0
            losses_train = []
            for x, ts, same, diff in train_data.batch(batch_size,
                                                      config.max_same,
                                                      config.max_diff,
                                                      config.output_shape):
                _, loss = train_model.get_loss_train(sess, x, ts, same, diff)
                losses_train.append(loss)
                if batch % config.log_interval == 0:
                    print("avg batch loss: %.4f" % np.mean(losses_train[-config.log_interval:]))
                # logger.info("batch {}, train_loss {}".format(batch, loss))
                batch += 1

            mean_loss_train = np.mean(losses_train)

            # use dev set for early stopping
            losses_dev = []
            for x, ts, same, diff in dev_data.batch(batch_size,
                                                    config.max_same,
                                                    config.max_diff,
                                                    config.output_shape):
                losses_dev.append(dev_model.get_loss_dev(sess, x, ts, same, diff))
            mean_loss_dev = np.mean(losses_dev)
            # print("avg dev loss: epoch {}, loss dev {}".format(epoch, mean_loss_dev))

            logger.info("epoch {}, train_loss {}, val_loss {}".format(epoch, mean_loss_train, mean_loss_dev))

            # early stopping
            if mean_loss_dev < best_loss:
                stopping_step = 0
                best_loss = mean_loss_dev
                # save the best model
                saver.save(sess, path.join(config.ckptdir, folder_model), global_step=epoch)
            else:
                stopping_step += 1

            if stopping_step >= config.early_stopping_step:
                break

            # # use dev set for showing the average precision of each epoch
            # embeddings, labels = [], []
            # for x, ts, ids in dev_data.batch(batch_size):
            #     embeddings.append(dev_model.get_embeddings(sess, x, ts))
            #     labels.append(ids)
            # embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
            # print("ap: %.4f" % average_precision(embeddings, labels))

if __name__ == "__main__":
    # margin = float(sys.argv[1])
    # mtl = sys.argv[2]
    # folder_model = sys.argv[3]

    margin = 0.3
    # output_shape list, [phn dimension, professionality dimension]
    # or integer 27 or 2
    output_shape = [27, 2]
    mtl = "phn"
    folder_model = 'model_cpu'
    num_layers = 1 if mtl == "pro" else 2

    main(margin_input=margin,
         output_shape=output_shape,
         num_layers=num_layers,
         mtl=mtl,
         folder_model=folder_model)
