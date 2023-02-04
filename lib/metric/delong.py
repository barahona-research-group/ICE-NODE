"""Two implementations for DeLong statistical test."""

from absl import logging
import numpy as np
import scipy.stats as st


class DeLongTest:
    """
    This class implements the statistical test of the difference of two AUCs
    from two different classifiers, starting from the ground truth
    and the predicted scores from each classifier. This class just serves
    as a namespace for the classmethod, no point to create instance objects
    from it.
    Thanks to Laksan Nathan for providing implementation to it in Python at:
        - https://biasedml.com/roc-comparison/
    Other resources:
        - Elizabeth DeLong et al. “Comparing the Areas under Two or
        More Correlated Receiver Operating
        Characteristic Curves: A Nonparametric Approach.” Biometrics 1988.
    """

    @classmethod
    def auc(cls, X, Y):
        """
        Compute the AUC using Mann-Whitney U statistic from the predicted scores
        of positve cases and negative cases given by X and Y, respectively.
        Args:
            X: the predicted scores for the positive cases, with shape (m,).
            Y: the predicted scores for the negative cases, with shape (n,).

        Returns:
            The AUC value estimated by Mann-Whitney U statistic.
        """
        m, n = len(X), len(Y)
        return sum([cls.kernel(x, y) for x in X for y in Y]) / (m * n)

    @staticmethod
    def kernel(X, Y):
        """https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test"""
        return .5 if Y == X else int(Y < X)

    @classmethod
    def structural_components(cls, X, Y):
        m, n = len(X), len(Y)
        V10 = [(1 / n) * sum([cls.kernel(x, y) for y in Y]) for x in X]
        V01 = [(1 / m) * sum([cls.kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    @staticmethod
    def get_S_entry(V_A, V_B, auc_A, auc_B):
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B)
                                         for a, b in zip(V_A, V_B)])

    @staticmethod
    def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB)**(.5))

    @staticmethod
    def group_preds_by_label(actual, preds):
        X = [p for (a, p) in zip(actual, preds) if a == 1]
        Y = [p for (a, p) in zip(actual, preds) if a == 0]
        return X, Y

    @classmethod
    def difference_test(cls, ground_truth, pred_scores_a, pred_scores_b):
        X_a, Y_a = cls.group_preds_by_label(ground_truth, pred_scores_a)
        X_b, Y_b = cls.group_preds_by_label(ground_truth, pred_scores_b)

        assert len(X_a) == len(X_b) and len(Y_a) == len(Y_b), "Unexpected!"
        m, n = len(X_a), len(Y_a)

        V_a10, V_a01 = cls.structural_components(X_a, Y_a)
        V_b10, V_b01 = cls.structural_components(X_b, Y_b)

        auc_a = cls.auc(X_a, Y_a)
        auc_b = cls.auc(X_b, Y_b)

        S_aa10 = cls.get_S_entry(V_a10, V_a10, auc_a, auc_a)
        S_aa01 = cls.get_S_entry(V_a01, V_a01, auc_a, auc_a)

        S_bb10 = cls.get_S_entry(V_b10, V_b10, auc_b, auc_b)
        S_bb01 = cls.get_S_entry(V_b01, V_b01, auc_b, auc_b)

        S_ab10 = cls.get_S_entry(V_a10, V_b10, auc_a, auc_b)
        S_ab01 = cls.get_S_entry(V_a01, V_b01, auc_a, auc_b)

        var_a = S_aa10 / m + S_aa01 / n
        var_b = S_bb10 / m + S_bb01 / n
        cov_ab = S_ab10 / m + S_ab01 / n

        # Two sided-test
        try:
            z = cls.z_score(var_a, var_b, cov_ab, auc_a, auc_b)
            p = st.norm.sf(abs(z)) * 2
        except ZeroDivisionError:
            logging.warning('Division by zero')
            p = float('nan')
        return auc_a, auc_b, var_a, var_b, p


class FastDeLongTest:
    """
    This class implements the statistical test of the difference of two AUCs
    from two different classifiers, starting from the ground truth
    and the predicted scores from each classifier. This class just serves
    as a namespace for the classmethod, no point to create instance objects
    from it.
    Thanks to Nikita Kazeev for providing implementation to it in Python at:
        - https://github.com/yandexdataschool/roc_comparison/blob/master/LICENSE
    Other resources:
        - Elizabeth DeLong et al. “Comparing the Areas under Two or
        More Correlated Receiver Operating
        Characteristic Curves: A Nonparametric Approach.” Biometrics 1988.
    """

    # AUC comparison adapted from
    # https://github.com/Netflix/vmaf/
    @staticmethod
    def compute_midrank(x):
        """Computes midranks.
        Args:
        x - a 1D numpy array
        Returns:
        array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    @classmethod
    def fastDeLong(cls, predictions_sorted_transposed, label_1_count):
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        Args:
        predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
            sorted such as the examples with label "1" are first
        Returns:
        (AUC value, DeLong covariance)
        Reference:
        @article{sun2014fast,
        title={Fast Implementation of DeLong's Algorithm for
                Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
        author={Xu Sun and Weichao Xu},
        journal={IEEE Signal Processing Letters},
        volume={21},
        number={11},
        pages={1389--1393},
        year={2014},
        publisher={IEEE}
        }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float)
        ty = np.empty([k, n], dtype=np.float)
        tz = np.empty([k, m + n], dtype=np.float)
        for r in range(k):
            tx[r, :] = cls.compute_midrank(positive_examples[r, :])
            ty[r, :] = cls.compute_midrank(negative_examples[r, :])
            tz[r, :] = cls.compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    @staticmethod
    def calc_pvalue(aucs, sigma):
        """Compute p-value of DeLong Test.
        Args:
        aucs: 1D array of AUCs
        sigma: AUC DeLong covariances
        Returns:
        log10(pvalue)
        """
        l = np.array([[1, -1]])
        z_num = np.abs(np.diff(aucs))
        z_den = np.sqrt(np.dot(np.dot(l, sigma), l.T))
        if (z_den == 0).any():
            logging.warning('Indeterminate test, p-value set to max (1.0)')
            return 1.0
        # Two sided-test
        z = z_num / z_den
        return st.norm.sf(abs(z.item())) * 2

    @staticmethod
    def compute_ground_truth_statistics(ground_truth):
        assert np.array_equal(np.unique(ground_truth), [0, 1])
        order = (-1 * ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        return order, label_1_count

    @classmethod
    def delong_roc_variance(cls, ground_truth, predictions):
        """
        Computes ROC AUC variance for a single set of predictions
        Args:
        ground_truth: np.array of 0 and 1
        predictions: np.array of floats of the probability of being class 1
        """
        order, label_1_count = cls.compute_ground_truth_statistics(
            ground_truth)
        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = cls.fastDeLong(predictions_sorted_transposed,
                                         label_1_count)
        assert len(
            aucs
        ) == 1, "There is a bug in the code, please forward this to the developers"
        return aucs[0], delongcov

    @classmethod
    def delong_roc_test(cls, ground_truth, predictions_one, predictions_two):
        """
        Computes log(p-value) for hypothesis that two ROC AUCs are different
        Args:
        ground_truth: np.array of 0 and 1
        predictions_one: predictions of the first model,
            np.array of floats of the probability of being class 1
        predictions_two: predictions of the second model,
            np.array of floats of the probability of being class 1
        """
        order, label_1_count = cls.compute_ground_truth_statistics(
            ground_truth)
        predictions_sorted_transposed = np.vstack(
            (predictions_one, predictions_two))[:, order]
        aucs, delongcov = cls.fastDeLong(predictions_sorted_transposed,
                                         label_1_count)
        p = cls.calc_pvalue(aucs, delongcov)

        return aucs[0], aucs[1], delongcov[0, 0], delongcov[1, 1], p
