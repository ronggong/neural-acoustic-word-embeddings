from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import pickle
import numpy as np
from collections import Counter
from sklearn import preprocessing


class Data_preprocessing(object):

    def __init__(self):
        self.list_feature_flatten = None
        self.label_integer = None
        self.le = preprocessing.LabelEncoder()
        self.scaler = None
        self.index_teacher = None
        self.index_student = None

    def feature_flatten(self, list_feature, list_key, data_str='_teacher'):
        """flatten the feature list"""
        list_feature_flatten = []
        list_key_flatten = []
        for ii in range(len(list_feature)):
            if list_key[ii] == u'?' or list_key[ii] == u'sil':
                pass
            else:
                list_feature_flatten += list_feature[ii]
                list_key_flatten += [list_key[ii] + data_str] * len(list_feature[ii])

        return list_feature_flatten, list_key_flatten

    def load_data_embedding_teacher_student(self,
                                            filename_feature_teacher,
                                            filename_feature_student,
                                            filename_list_key_teacher,
                                            filename_list_key_student,
                                            filename_scaler):
        """
        load data for the RNN phone embedding model
        """
        list_feature_teacher = pickle.load(open(filename_feature_teacher, 'rb'))
        list_key_teacher = pickle.load(open(filename_list_key_teacher, 'rb'))
        list_feature_student = pickle.load(open(filename_feature_student, 'rb'))
        list_key_student = pickle.load(open(filename_list_key_student, 'rb'))
        self.scaler = pickle.load(open(filename_scaler, 'rb'))

        # flatten the feature and label list
        list_feature_flatten_teacher, list_key_flatten_teacher = \
            self.feature_flatten(list_feature_teacher, list_key_teacher, '_teacher')
        list_feature_flatten_student, list_key_flatten_student = \
            self.feature_flatten(list_feature_student, list_key_student, '_student')

        self.list_feature_flatten = list_feature_flatten_teacher + list_feature_flatten_student
        list_key_flatten = list_key_flatten_teacher + list_key_flatten_student

        # encode the label to integer
        self.le.fit(list_key_flatten)
        self.label_integer = self.le.transform(list_key_flatten)

    def load_training_data(self, config):
        """
        load training data, data transform
        """
        self.load_data_embedding_teacher_student(filename_feature_teacher=config.filename_feature_teacher,
                                                 filename_feature_student=config.filename_feature_student,
                                                 filename_list_key_teacher=config.filename_list_key_teacher,
                                                 filename_list_key_student=config.filename_list_key_student,
                                                 filename_scaler=config.filename_scaler)

    def load_val_data(self, config):
        """
        load validation data, data transform
        """
        self.load_data_embedding_teacher_student(filename_feature_teacher=config.filename_feature_teacher_val,
                                                 filename_feature_student=config.filename_feature_student_val,
                                                 filename_list_key_teacher=config.filename_list_key_teacher,
                                                 filename_list_key_student=config.filename_list_key_student,
                                                 filename_scaler=config.filename_scaler)

    def load_test_data(self, config):
        """
        load test data, data transform
        """
        self.load_data_embedding_teacher_student(filename_feature_teacher=config.filename_feature_teacher_test,
                                                 filename_feature_student=config.filename_feature_student_test,
                                                 filename_list_key_teacher=config.filename_list_key_teacher,
                                                 filename_list_key_student=config.filename_list_key_student,
                                                 filename_scaler=config.filename_scaler)

    def organize_label(self, config):
        """
        organize label for output shape 2 (professionality) or 27 (phoneme)
        """
        # transform the labels
        labels = self.le.inverse_transform(self.label_integer)
        if config.output_shape == 2:
            indices_teacher = [i for i, s in enumerate(labels) if 'teacher' in s]
            indices_student = [i for i, s in enumerate(labels) if 'student' in s]
            self.label_integer[indices_teacher] = 0
            self.label_integer[indices_student] = 1
        elif config.output_shape == 27:
            phn_set = list(set([l.split('_')[0] for l in labels]))
            for ii in range(len(phn_set)):
                indices_phn = [i for i, s in enumerate(labels) if phn_set[ii] == s.split('_')[0]]
                self.label_integer[indices_phn] = ii
            self.index_teacher = [ii for ii in range(len(self.label_integer)) if 'teacher' in labels[ii]]
            self.index_student = [ii for ii in range(len(self.label_integer)) if 'student' in labels[ii]]
        else:
            pass

    def process_train(self, config):

        self.load_training_data(config=config)

        self.organize_label(config=config)

        # split into train and validation index
        train_index, val_index = pickle.load(open(config.filename_data_splits, 'rb'))

        list_feature_fold_train = [self.scaler.transform(self.list_feature_flatten[ii]) for ii in train_index]
        labels_integer_fold_train = self.label_integer[train_index]
        # list_feature_fold_train = [np.expand_dims(feature, axis=0) for feature in list_feature_fold_train]

        list_feature_fold_val = [self.scaler.transform(self.list_feature_flatten[ii]) for ii in val_index]
        labels_integer_fold_val = self.label_integer[val_index]
        # list_feature_fold_val = [np.expand_dims(feature, axis=0) for feature in list_feature_fold_val]

        return list_feature_fold_train, labels_integer_fold_train, list_feature_fold_val, labels_integer_fold_val

    def process_val(self, config):
        self.load_val_data(config=config)
        self.organize_label(config=config)

    def process_test(self, config):
        self.load_test_data(config=config)
        self.organize_label(config=config)


class Dataset(object):
    """Creat data class."""

    def __init__(self, data, labels, partition, config):
        """Initialize dataset."""
        self.is_test = (partition == "test")

        self.feature_dim = config.feature_dim

        # data_scp = getattr(config, "%sfile" % partition)
        # labels, data = zip(*read_mat_scp(data_scp))
        #
        # words = [re.split("_", x)[0] for x in labels]
        # uwords = np.unique(words)
        #
        # word2id = {v: k for k, v in enumerate(uwords)}
        # ids = [word2id[w] for w in words]
        #
        # feature_mean, n = 0.0, 0
        # for x in data:
        #     feature_mean += np.sum(x)
        #     n += np.prod(x.shape)
        # self.feature_mean = feature_mean / n

        self.data = data
        self.ids = np.array(labels, dtype=np.int32)
        self.id_counts = Counter(labels)

        self.num_classes = len(self.id_counts)
        self.num_examples = len(self.ids)

    def shuffle(self):
        """Shuffle data."""

        shuffled_indices = np.random.permutation(self.num_examples)
        self.data = [self.data[ii] for ii in shuffled_indices]
        self.ids = self.ids[shuffled_indices]

    def pad_features(self, indices):
        """Pad acoustic features to max length sequence."""
        b = len(indices)
        data = [self.data[ind] for ind in indices]
        lens = np.array([len(xx) for xx in data], dtype=np.int32)
        padded = np.zeros((b, max(lens), self.feature_dim))
        for i, (x, l) in enumerate(zip(data, lens)):
            padded[i, :l] = x

        return padded, lens, self.ids[indices]

    def gen_diff(self, diff_ids):
        """gen diff index matrix from diff_ids, the latter is the diff label matrix"""
        diff = np.full_like(diff_ids, 0, dtype=np.int32)
        for label, count in self.id_counts.items():  # collect diff samples
            # indices are where we have the label
            indices = np.where(diff_ids == label)
            # diff stores the indices of the features which has the label
            diff[indices] = np.where(self.ids == label)[0][np.random.randint(0, count, len(indices[0]))]
        return diff

    def batch(self, batch_size, max_same=1, max_diff=1, output_shape=2):
        """Batch data."""

        self.shuffle()

        # phn same, pro same
        same = []
        for index, label in enumerate(self.ids):  # collect same samples
            indices = np.where(self.ids == label)[0]
            same.append(np.random.permutation(indices[indices != index])[:max_same])
        same = np.array(same)

        if isinstance(output_shape, list):
            diff_ids_phn_pro = np.zeros((self.num_examples, max_diff), dtype=np.int32)
            diff_ids_pro = np.zeros((self.num_examples, max_diff), dtype=np.int32)
            diff_ids_phn = np.zeros((self.num_examples, max_diff), dtype=np.int32)
            ids_odd = np.arange(1, self.num_classes, 2, dtype=np.int32)  # teacher index
            ids_even = np.arange(0, self.num_classes, 2, dtype=np.int32)  # student index
            for ii_example in range(self.num_examples):
                # phn same, pro diff
                # if ii_example is even, it's student. Its teacher id = its id+1
                # the diff ids need to be id+1
                # if ii_example is odd, it's teacher. Its student id = its id-1
                diff_ids_example = np.array([self.ids[ii_example] + 1]) \
                    if self.ids[ii_example] % 2 == 0 else \
                    np.array([self.ids[ii_example] - 1])
                diff_ids_pro[ii_example, :] = np.random.choice(diff_ids_example, size=max_diff)

                # phn diff, pro same
                # if id is even, remove the same id from even id list, or vice versa
                diff_ids_example = np.delete(ids_even, np.where(ids_even == self.ids[ii_example])) \
                    if self.ids[ii_example] % 2 == 0 else \
                    np.delete(ids_odd, np.where(ids_odd == self.ids[ii_example]))
                diff_ids_phn[ii_example, :] = np.random.choice(diff_ids_example, size=max_diff)

                # phn diff, pro diff
                # if ii_example is even, it's student. Its teacher id = its id+1
                # the diff ids need to select from the teacher ids and remove its id+1
                diff_ids_example = np.delete(ids_odd, np.where(ids_odd == self.ids[ii_example]+1)) \
                    if self.ids[ii_example] % 2 == 0 else \
                    np.delete(ids_even, np.where(ids_even == self.ids[ii_example]-1))
                diff_ids_phn_pro[ii_example, :] = np.random.choice(diff_ids_example, size=max_diff)
        else:
            raise ValueError('output shape needs to be a list, this is a joint learning case')

        # diff example index
        diff_pro = self.gen_diff(diff_ids=diff_ids_pro)
        diff_phn = self.gen_diff(diff_ids=diff_ids_phn)
        diff_phn_pro = self.gen_diff(diff_ids=diff_ids_phn_pro)

        get_batch_indices = lambda start: range(start, min(start + batch_size, self.num_examples))

        for indices in map(get_batch_indices, range(0, self.num_examples, batch_size)):

            if self.is_test:
                yield self.pad_features(indices)
            else:
                b = len(indices)

                same_partition = [np.arange(b)]  # same segment ids for anchors
                same_partition += [(b + i) * np.ones(len(x)) for i, x in
                                   enumerate(same[indices])]  # same segment ids for same examples
                same_partition += [(2 * b) + np.arange(max_diff * b)]  # same segment ids for diff examples
                same_partition = np.concatenate(same_partition)

                diff_partition = np.concatenate(
                    [i * np.ones(max_diff) for i in range(b)])  # diff segment ids for diff examples

                indices = np.concatenate((indices,
                                          np.hstack(same[indices]),
                                          diff_pro[indices].flatten(),
                                          diff_phn[indices].flatten(),
                                          diff_phn_pro[indices].flatten()))

                data, lens, _ = self.pad_features(indices)
                yield data, lens, same_partition, diff_partition
