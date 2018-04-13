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
        dist_mat = (2.0 - squareform(pdist(embeddings, 'cosine'))) / 2.0
        gt_mat = self.ground_truth_matrix(labels)

        # we only compare teacher to student embeddings
        dist_mat = dist_mat[:min(index_student), min(index_student):]
        gt_mat = gt_mat[:min(index_student), min(index_student):]

        ap = self.eval_embeddings_no_trim(dist_mat=dist_mat, gt_mat=gt_mat)

        return ap

