import os
import numpy as np
import abc
import pandas as pd


class FeatureSelection:
    """
    Parent class for all feature selection methods (filter, wrapper and heuristic)

    Args:
        name (str): Results folder name
        target (str): Target feature name
        model (list): List of sklearn learning method objects
        train (pd.DataFrame): Training data
        test (pd.DataFrame, None): Testing data
        drops (list): Features to drop before execution
        metric (str): Metric to optimize between accuracy, precision and recall
        Tmax (int): Total number of seconds allocated before shutdown
        ratio (float): Importance of the number of features selected in relation to the score calculated
        suffix (str): Suffix in the folder name of a method (Important when lauching twice the same method !)
    """
    def __init__(self, name, target, model, train, test=None, drops=None, metric=None, Tmax=None, ratio=None,
                 suffix=None):
        drops = drops or []
        self.train = train.drop(drops, axis=1)
        if isinstance(test, pd.DataFrame):
            self.test = test.drop(drops, axis=1)
        else:
            self.test = None
        self.name = name
        self.target = target
        self.cols = self.train.drop([target], axis=1).columns
        unique, count = np.unique(train[target], return_counts=True)
        self.n_class = len(unique)
        self.D = len(self.cols)
        self.metric = metric or "accuracy"
        self.model = model
        self.Tmax = Tmax or 3600
        self.ratio = ratio or 0.001
        self.suffix = suffix or ''
        self.path = os.path.join(os.getcwd(), os.path.join('out', self.name))
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    @abc.abstractmethod
    def start(self, pid, result_queue):
        pass

    @staticmethod
    def evaluate_confusion_matrix(confusion_matrix):
        nb_classes = confusion_matrix.shape[0]
        results = []
        for i in range(nb_classes):
            tp = confusion_matrix[i, i]
            tn = np.sum(confusion_matrix) - np.sum(confusion_matrix[i, :]) - np.sum(confusion_matrix[:, i]) + tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            fp = np.sum(confusion_matrix[:, i]) - tp
            total = np.sum(confusion_matrix[i, :])
            results.append([tp, tn, fp, fn, total])
        return results

