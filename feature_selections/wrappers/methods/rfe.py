import os
import time
import numpy as np

from feature_selections.wrappers.wrapper import Wrapper
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector
from datetime import timedelta
from utility import createDirectory, fitness_ind_models


class Rfe(Wrapper):
    def __init__(self, name, target, model, train, test=None, drops=None, metric=None, nexp=None, Tmax=None,
                 ratio=None, suffix=None, k=None, step=None):
        super().__init__(name, target, model, train, test, drops, metric, nexp, Tmax, ratio, suffix, k, step)
        """
        Class that implements foward feature elimination if k<(D/2) or backward feature elimination otherwise
        """
        self.path = os.path.join(self.path, 'rfe' + self.suffix)
        createDirectory(path=self.path)

    def start(self, pid, result_queue):
        name = "Recursive Feature Elimination"
        debut = time.time()
        old_path = self.path
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        score, model, col, vector = 0, [], [], []
        same, stop = 0, False
        time_debut = timedelta(seconds=(time.time() - debut))
        for i in range(len(self.model)):
            instant = time.time()
            same = same + 1
            try:
                self.train.columns = self.train.columns.astype(str)
                self.test.columns = self.test.columns.astype(str)
            except ValueError:
                pass
            X, y = self.train.drop(self.target, axis=1), self.train[self.target]
            try:
                rfe = RFE(self.model[i], n_features_to_select=self.k, step=1)
                rfe.fit(X, y)
                top_k_features = [f for f, s in zip(X.columns, rfe.support_) if s]
            except ValueError:
                if self.k < int(self.D/2):
                    rfe = SequentialFeatureSelector(self.model[i], k_features=self.k, forward=True)
                else:
                    rfe = SequentialFeatureSelector(self.model[i], k_features=self.k, forward=False)
                rfe.fit(X, y)
                top_k_features = [x for x in list(rfe.k_feature_names_)]
            v = []
            for c in X.columns:
                if c in top_k_features:
                    v.append(1)
                else:
                    v.append(0)
            v.append(i)
            s, m, c = fitness_ind_models(train=self.train, test=self.test, ind=v, target_name=self.target,
                                         metric=self.metric, model=self.model[v[-1]], ratio=self.ratio)
            if s > score:
                same = 0
                score, model, col, vector = s, m, c, v
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            print_out = self.print_(print_out=print_out, name="RFE ", maxi=score, best=s, mean=s,
                                    worst=s, feats=len(col), time_exe=time_instant,
                                    time_total=time_debut, g=i, cpt=same) + "\n"
            if time.time() - debut >= self.Tmax:
                stop = True
            if i % 10 == 0 or stop:
                self.write(name=name, colMax=col, bestScore=score, bestModel=model, bestInd=vector, g=len(self.model),
                           t=timedelta(seconds=(time.time() - debut)), last=len(self.model) - same, out=print_out)
                print_out = ""
                if stop:
                    break
        self.write(name=name, colMax=col, bestScore=score, bestModel=model, bestInd=vector, g=len(self.model),
                   t=timedelta(seconds=(time.time() - debut)), last=len(self.model) - same, out=print_out)
        result_queue.put((pid, score, time_debut, len(self.model) - same, len(col), old_path))
