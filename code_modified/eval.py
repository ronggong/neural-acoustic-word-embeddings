from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
import os
import csv
from config import Config
import numpy as np
import tensorflow as tf
from model import Model
from data import Data_preprocessing
from data import Dataset
import pandas as pd
from average_precision import Ap_score


def main(margin_input, folder_model, num_layers, output_shape, val_test):

    config = Config(margin_input=margin_input,
                    folder_model=folder_model,
                    num_layers=num_layers,
                    output_shape=output_shape)

    DATA_PREPROCESSING = Data_preprocessing()

    if val_test == 'val':
        DATA_PREPROCESSING.load_val_data(config=config)
    elif val_test == 'test':
        DATA_PREPROCESSING.load_test_data(config=config)
    else:
        raise ValueError('{} doesn''t exist.'.format(val_test))

    if output_shape == 27:
        DATA_PREPROCESSING.organize_label(config=config)

    AP_SCORE = Ap_score()

    val_data = Dataset(data=DATA_PREPROCESSING.list_feature_flatten,
                       labels=DATA_PREPROCESSING.label_integer,
                       partition="test",
                       config=config)

    # train_model = Model(is_train="train", config=config, reuse=False)
    val_model = Model(is_train="test", config=config, reuse=tf.AUTO_REUSE)

    # batch_size = config.batch_size
    batch_size = 1
    list_ap = []
    array_ap_phn_5_runs = np.zeros((5, 27))
    for ii in range(5):
        print("{} margin, evaluate run time: {}".format(margin_input, ii+1))

        saver = tf.train.Saver()

        proto = tf.ConfigProto(intra_op_parallelism_threads=0)
        with tf.Session(config=proto) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(config.ckptdir+str(ii+1))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("restored from %s" % ckpt.model_checkpoint_path)

            # use val set for showing the average precision of each epoch
            embeddings, labels = [], []
            for x, ts, ids in val_data.batch(batch_size):
                embeddings.append(val_model.get_embeddings(sess, x, ts))
                labels.append(ids)
            embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)

            if output_shape == 27:
                ap = AP_SCORE.eval_pronunciation(embeddings=embeddings,
                                                 labels=labels,
                                                 index_student=DATA_PREPROCESSING.index_student)
            elif output_shape == 2:
                ap, array_ap_phn, list_ratio_tea_stu, cols = AP_SCORE.eval_professionality(embeddings=embeddings,
                                                                                           labels=labels,
                                                                                           le=DATA_PREPROCESSING.le)
                array_ap_phn_5_runs[ii, :] = array_ap_phn
            else:
                pass
            list_ap.append(ap)
            # print("ap: %.4f" % average_precision(embeddings, labels))
    print(margin_input, np.mean(list_ap), np.std(list_ap))

    with open(os.path.join('./results_eval/',
                           val_test+'_'+str(output_shape)+'_'+str(margin_input)+'.csv'), 'w') as csvfile:
        results_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow([margin_input, np.mean(list_ap), np.std(list_ap)])

    if val_test == 'test' and output_shape == 2:
        # organize the Dataframe
        ap_phn_mean = np.mean(array_ap_phn_5_runs, axis=0)
        ap_phn_std = np.std(array_ap_phn_5_runs, axis=0)
        ap_phn_mean_std = pd.DataFrame(np.transpose(np.vstack((ap_phn_mean, ap_phn_std, list_ratio_tea_stu))),
                                       columns=['mean', 'std', 'ratio'],
                                       index=cols)

        ap_phn_mean_std = ap_phn_mean_std.sort_values(by='mean')
        ap_phn_mean_std.to_csv(os.path.join('./results_eval/',
                                            val_test+'_'+str(output_shape)+'_'+str(margin_input)+'_phn_mean_std.csv'))


if __name__ == "__main__":
    # margin = float(sys.argv[1])
    # folder_model = sys.argv[2]

    # val_test = 'val'
    # output_shape = 2
    # for margin_str in ['015', '0', '03', '045', '06']:
    #     if margin_str == '0':
    #         margin = 0.0
    #     elif margin_str == '015':
    #         margin = 0.15
    #     elif margin_str == '03':
    #         margin = 0.3
    #     elif margin_str == '045':
    #         margin = 0.45
    #     elif margin_str == '06':
    #         margin = 0.6
    #     else:
    #         raise ValueError('{} doesn''t exist.'.format(margin_str))
    #
    #     folder_model = 'model_gpu_'+str(output_shape)+'_'+margin_str+'_'
    #
    #     num_layers = 2 if output_shape == 27 else 1
    #
    #     main(margin_input=margin,
    #          folder_model=folder_model,
    #          num_layers=num_layers,
    #          output_shape=output_shape,
    #          val_test=val_test)

    val_test = 'test'
    output_shape = 2
    margin_str = '015'
    num_layers = 1
    folder_model = 'model_gpu_' + str(output_shape) + '_' + margin_str + '_'
    main(margin_input=0.15,
         folder_model=folder_model,
         num_layers=num_layers,
         output_shape=output_shape,
         val_test=val_test)
