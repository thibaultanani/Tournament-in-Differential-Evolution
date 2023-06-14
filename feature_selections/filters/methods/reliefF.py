import os
import time
import numpy as np

from feature_selections.filters.filter import Filter
from datetime import timedelta
from skrebate import ReliefF as reliefF
from utility import createDirectory, fitness_ind_models


class ReliefF(Filter):
    """
    Class that uses ReliefF as a filter method
    """
    def __init__(self, name, target, model, train, test=None, drops=None, metric=None, Tmax=None, ratio=None,
                 suffix=None, k=None):
        super().__init__(name, target, model, train, test, drops, metric, Tmax, ratio, suffix, k)
        self.path = os.path.join(self.path, 'relieff' + self.suffix)
        createDirectory(path=self.path)

    def start(self, pid):
        name = "ReliefF"
        debut = time.time()
        old_path = self.path
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        X = self.train.drop([self.target], axis=1).values
        y = self.train[self.target].values
        relief = reliefF(n_features_to_select=self.k)
        relief.fit(X, y)
        importances = relief.feature_importances_
        top_k_features = self.cols[importances.argsort()[-self.k:][::-1]]
        score, model, col, vector = 0, [], [], []
        same, stop = 0, False
        time_debut = timedelta(seconds=(time.time() - debut))
        for i in range(len(self.model)):
            instant = time.time()
            same = same + 1
            v = [0] * self.D
            for var in top_k_features:
                v[self.cols.get_loc(var)] = 1
            v.append(i)
            s, m, c = fitness_ind_models(train=self.train, test=self.test, ind=v, target_name=self.target,
                                         metric=self.metric, model=self.model[v[-1]], ratio=self.ratio)
            if s > score:
                same = 0
                score, model, col, vector = s, m, c, v
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            print_out = self.print_(print_out=print_out, name="RELI", maxi=score, best=s, mean=s,
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
