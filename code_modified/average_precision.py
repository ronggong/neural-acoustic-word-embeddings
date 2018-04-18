from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
from scipy.misc import comb
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics import average_precision_score


def average_precision(data, labels):
    """
    Calculate average precision and precision-recall breakeven, and return
    the average precision / precision-recall breakeven calculated
    using `same_dists` and `diff_dists`.
    -------------------------------------------------------------------
    returns average_precision, precision-recall break even : (float, float)
    """
    num_examples = len(labels)
    num_pairs = int(comb(num_examples, 2))

    # build up binary array of matching examples
    matches = np.zeros(num_pairs, dtype=np.bool)

    i = 0
    for n in range(num_examples):
        j = i + num_examples - n - 1
        matches[i:j] = (labels[n] == labels[n + 1:]).astype(np.int32)
        i = j

    num_same = np.sum(matches)

    # calculate pairwise distances and sort matches
    dists = pdist(data, metric="cosine")
    matches = matches[np.argsort(dists)]

    # calculate precision, average precision, and recall
    precision = np.cumsum(matches) / np.arange(1, num_pairs + 1)
    average_precision = np.sum(precision * matches) / num_same
    recall = np.cumsum(matches) / num_same

    # multiple precisions can be at single recall point, take max
    for n in range(num_pairs - 2, -1, -1):
        precision[n] = max(precision[n], precision[n + 1])

    # calculate precision-recall breakeven
    prb_ix = np.argmin(np.abs(recall - precision))
    prb = (recall[prb_ix] + precision[prb_ix]) / 2.

    return average_precision


class Ap_score(object):

    @staticmethod
    def ground_truth_matrix(y_test):
        """
        ground truth mat
        :param y_test:
        :return:
        """
        sample_num = len(y_test)

        gt_matrix = np.zeros((sample_num, sample_num))

        for ii in range(sample_num - 1):
            for jj in range(ii + 1, sample_num):
                if y_test[ii] == y_test[jj]:
                    gt_matrix[ii, jj] = 1.0
                else:
                    gt_matrix[ii, jj] = 0.0
        return gt_matrix

    @staticmethod
    def eval_embeddings_no_trim(dist_mat, gt_mat):
        """
        average precision score
        :param dist_mat:
        :param gt_mat:
        :return:
        """
        assert dist_mat.shape == gt_mat.shape
        ap = average_precision_score(y_true=np.squeeze(np.abs(gt_mat)),
                                     y_score=np.squeeze(np.abs(dist_mat)),
                                     average='weighted')
        return ap

    def eval_pronunciation(self, embeddings, labels, index_student):
        """pronunciation evaluation, only consider teacher to student"""
        dist_mat = (2.0 - squareform(pdist(embeddings, 'cosine'))) / 2.0
        gt_mat = self.ground_truth_matrix(labels)

        # we only compare teacher to student embeddings
        dist_mat = dist_mat[:min(index_student), min(index_student):]
        gt_mat = gt_mat[:min(index_student), min(index_student):]

        ap = self.eval_embeddings_no_trim(dist_mat=dist_mat, gt_mat=gt_mat)

        return ap

    def eval_professionality(self, embeddings, labels, le):
        list_dist = []
        list_gt = []
        array_ap_phn = np.zeros((27,))
        cols = []
        list_ratio_tea_stu = []
        for ii_class in range(27):
            # teacher student pair class index
            idx_ii_class = np.where(np.logical_or(labels == 2 * ii_class,
                                                  labels == 2 * ii_class + 1))[0]

            idx_ii_class_stu = len(np.where(labels == 2 * ii_class)[0])
            idx_ii_class_tea = len(np.where(labels == 2 * ii_class + 1)[0])

            # ratio of teacher's samples
            list_ratio_tea_stu.append(idx_ii_class_tea / float(idx_ii_class_tea + idx_ii_class_stu))

            dist_mat = (2.0 - squareform(pdist(embeddings[idx_ii_class], 'cosine'))) / 2.0
            labels_ii_class = [labels[idx] for idx in idx_ii_class]
            gt_mat = self.ground_truth_matrix(labels_ii_class)

            sample_num = dist_mat.shape[0]
            iu1 = np.triu_indices(sample_num, 1)  # trim the upper mat

            list_dist.append(dist_mat[iu1])
            list_gt.append(gt_mat[iu1])

            # calculate the average precision of each phoneme
            ap_phn = average_precision_score(y_true=np.abs(list_gt[ii_class]),
                                             y_score=np.abs(list_dist[ii_class]),
                                             average='weighted')

            cols.append(le.inverse_transform(2 * ii_class).split('_')[0])
            array_ap_phn[ii_class] = ap_phn

            print(list_ratio_tea_stu)

        array_dist = np.concatenate(list_dist)
        array_gt = np.concatenate(list_gt)

        ap = average_precision_score(y_true=np.abs(array_gt),
                                     y_score=np.abs(array_dist),
                                     average='weighted')

        return ap, array_ap_phn, list_ratio_tea_stu, cols
