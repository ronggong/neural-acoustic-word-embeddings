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


def main(margin_input, folder_model, num_layers, output_shape, val_test, mtl, joint):

    config = Config(margin_input=margin_input,
                    folder_model=folder_model,
                    num_layers=num_layers,
                    output_shape=output_shape,
                    mtl=mtl)

    DATA_PREPROCESSING = Data_preprocessing()

    if val_test == 'val':
        DATA_PREPROCESSING.load_val_data(config=config)
    elif val_test == 'test':
        DATA_PREPROCESSING.load_test_data(config=config)
    else:
        raise ValueError('{} doesn''t exist.'.format(val_test))

    if mtl == 'phn':
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
            embeddings_27, embeddings_2, labels = [], [], []
            for x, ts, ids in val_data.batch(batch_size, output_shape=output_shape):
                embeddings_phn, embeddings_pro = val_model.get_embeddings(sess, x, ts)
                embeddings_27.append(embeddings_phn)
                embeddings_2.append(embeddings_pro)
                labels.append(ids)
            embeddings_27, embeddings_2, labels = np.concatenate(embeddings_27), \
                                                  np.concatenate(embeddings_2), \
                                                  np.concatenate(labels)

            if mtl == "phn":
                ap = AP_SCORE.eval_pronunciation(embeddings=embeddings_27,
                                                 labels=labels,
                                                 index_student=DATA_PREPROCESSING.index_student)
            elif mtl == "pro":
                ap, array_ap_phn, list_ratio_tea_stu, cols = AP_SCORE.eval_professionality(embeddings=embeddings_2,
                                                                                           labels=labels,
                                                                                           le=DATA_PREPROCESSING.le)
                array_ap_phn_5_runs[ii, :] = array_ap_phn
            else:
                pass
            list_ap.append(ap)
            # print("ap: %.4f" % average_precision(embeddings, labels))
    print(margin_input, np.mean(list_ap), np.std(list_ap))

    joint_str = 'both_' if joint else ''

    with open(os.path.join('./results_eval/',
                           val_test+'_mtl_'+joint_str+mtl+'_'+str(output_shape)+'_'+str(margin_input)+'.csv'), 'w') as csvfile:
        results_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow([margin_input, np.mean(list_ap), np.std(list_ap)])

    if val_test == 'test' and mtl == 'pro':
        # organize the Dataframe
        ap_phn_mean = np.mean(array_ap_phn_5_runs, axis=0)
        ap_phn_std = np.std(array_ap_phn_5_runs, axis=0)
        ap_phn_mean_std = pd.DataFrame(np.transpose(np.vstack((ap_phn_mean, ap_phn_std, list_ratio_tea_stu))),
                                       columns=['mean', 'std', 'ratio'],
                                       index=cols)

        ap_phn_mean_std = ap_phn_mean_std.sort_values(by='mean')
        ap_phn_mean_std.to_csv(os.path.join('./results_eval/',
                                            val_test+'_mtl_'+joint_str+mtl+'_'+str(output_shape)+'_'+str(margin_input)+'_phn_mean_std.csv'))


if __name__ == "__main__":
    # margin = float(sys.argv[1])
    # folder_model = sys.argv[2]

    val_test = 'val'
    output_shape = [27, 2]
    joint = True
    mtl = 'pro'
    for margin_str in ['015', '03', '045', '06']:
        if margin_str == '0':
            margin = 0.0
        elif margin_str == '015':
            margin = 0.15
        elif margin_str == '03':
            margin = 0.3
        elif margin_str == '045':
            margin = 0.45
        elif margin_str == '06':
            margin = 0.6
        else:
            raise ValueError('{} doesn''t exist.'.format(margin_str))

        if joint:
            folder_model = 'model_gpu_mtl_both_'+margin_str+'_'
        else:
            folder_model = 'model_gpu_mtl_'+mtl+'_'+margin_str+'_'

        num_layers = 1 if (mtl == 'pro' and not joint) else 2

        main(margin_input=margin,
             folder_model=folder_model,
             num_layers=num_layers,
             output_shape=output_shape,
             val_test=val_test,
             mtl=mtl,
             joint=joint)

    # val_test = 'test'
    # output_shape = [27, 2]
    # margin_str = '045'
    # mtl = 'pro'
    # num_layers = 1 if mtl == 'pro' else 2
    # folder_model = 'model_gpu_mtl_'+mtl+'_'+margin_str+'_'
    # main(margin_input=0.45,
    #      folder_model=folder_model,
    #      num_layers=num_layers,
    #      output_shape=output_shape,
    #      val_test=val_test,
    #      mtl=mtl)
