import abc
import os
import psutil

from feature_selections import FeatureSelection


class Filter(FeatureSelection):
    """
    Parent class for filter methods

    Args:
        k (int): Number of features to select
    """
    def __init__(self, name, target, model, train, test=None, drops=None, metric=None, Tmax=None,
                 ratio=None, suffix=None, k=None):
        super().__init__(name, target, model, train, test, drops, metric, Tmax, ratio, suffix)
        self.k = k or int(self.D/2)

    @abc.abstractmethod
    def start(self, pid, result_queue):
        pass

    @staticmethod
    def print_(print_out, name, maxi, best, mean, worst, feats, time_exe, time_total, g, cpt):
        display = "[{}]    PID: [{:3}]    G: {:5d}    max: {:2.4%}    features: {:6d}    best: {:2.4%}" \
                  "   mean: {:2.4%}    worst: {:2.4%}    G time: {}    T time: {}    last: {:6d}" \
            .format(name, 1, g, maxi, feats, best, mean, worst, time_exe, time_total, cpt)
        print_out = print_out + display
        print(display)
        return print_out

    def write(self, name, colMax, bestScore, bestModel, bestInd, g, t, last, out):
        a = os.path.join(os.path.join(self.path, 'results.txt'))
        f = open(a, "w")
        methods = [self.model[m].__class__.__name__ for m in range(len(self.model))]
        matrix_str = ''
        for i, res in enumerate(self.evaluate_confusion_matrix(bestModel)):
            matrix_str = matrix_str + "Class: " + str(i) + " TP: " + str(res[0]) + " TN: " + str(res[1]) + \
                         " FP: " + str(res[2]) + " FN: " + str(res[3]) + " Total: " + str(res[4]) + "\n"
        string = "Filter: " + name + os.linesep + \
                 "Iterations: " + str(g) + os.linesep + \
                 "Latest Improvement: " + str(last) + os.linesep + \
                 "k: " + str(self.k) + os.linesep + \
                 "Methods List: " + str(methods) + os.linesep + \
                 "Best Method: " + self.model[bestInd[-1]].__class__.__name__ + os.linesep + \
                 "Best Score: " + str(bestScore) + os.linesep + \
                 "Best Model: " + os.linesep + matrix_str + "\n" + \
                 "Best Subset: " + str(bestInd) + os.linesep + "Columns: " + str(colMax) + os.linesep + \
                 "Number of Features: " + str(len(colMax)) + os.linesep + \
                 "Execution Time: " + str(round(t.total_seconds())) + " (" + str(t) + ")" + os.linesep + \
                 "Memory: " + str(psutil.virtual_memory())
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path, 'log.txt'))
        f = open(a, "a")
        f.write(out)
