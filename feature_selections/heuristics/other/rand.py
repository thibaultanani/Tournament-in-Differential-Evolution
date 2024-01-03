import os
import time
import numpy as np
import warnings

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility.utility import createDirectory, add, fitness_models

warnings.filterwarnings('ignore')


class Random(Heuristic):
    """
    Class that implements the random search heuristic
    """
    def __init__(self, name, target, train, test, model, drops=None, metric=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, suffix=None):
        super().__init__(name, target, model, train, test, drops, metric, Tmax, ratio, suffix, N, Gmax)
        self.path = os.path.join(self.path, 'rand' + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def print_(print_out, pid, maxi, best, mean, worst, feats, time_exe, time_total, g, cpt):
        display = "[RAND]    PID: [{:3}]    G: {:5d}    max: {:2.4%}    features: {:6d}    best: {:2.4%}" \
                  "   mean: {:2.4%}    worst: {:2.4%}    G time: {}    T time: {}    last: {:6d}" \
            .format(pid, g, maxi, feats, best, mean, worst, time_exe, time_total, cpt)
        print_out = print_out + display
        print(display)
        return print_out

    def specifics(self, colMax, bestScore, bestModel, bestInd, g, t, last, out):
        self.write("Random Generation", colMax, bestScore, bestModel, bestInd, g, t, last, "", out)

    @staticmethod
    def create_population_models(inds, size, models):
        # Initialise the population
        pop = np.zeros((inds, size), dtype=bool)
        for i in range(inds):
            num_true = np.random.randint(1, size)
            true_indices = np.random.choice(size - 1, size=num_true, replace=False)
            pop[i, true_indices] = True
        pop = pop.astype(int)
        # Replace last element with random integer between 0 and models-1
        pop[:, -1] = np.random.randint(0, len(models), size=inds)
        return pop

    def start(self, pid):
        code = "RAND"
        debut = time.time()
        old_path = self.path
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        # Measuring the execution time
        instant = time.time()
        # Generation (G) initialisation
        G, same, stop = 0, 0, False
        # Population P initialisation
        P = self.create_population_models(inds=self.N, size=self.D + 1, models=self.model)
        # Evaluates population
        scores, models, cols = fitness_models(train=self.train, test=self.test, pop=P, target_name=self.target,
                                              metric=self.metric, model=self.model, ratio=self.ratio)
        bestScore, worstScore, bestModel, bestInd, bestCols = \
            add(scores=scores, models=models, inds=np.asarray(P), cols=cols)
        scoreMax, modelMax, indMax, colMax = bestScore, bestModel, bestInd, bestCols
        mean_scores = float(np.mean(scores))
        time_instant = timedelta(seconds=(time.time() - instant))
        time_debut = timedelta(seconds=(time.time() - debut))
        # Pretty print the results
        print_out = self.sprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                 mean=mean_scores, worst=worstScore, feats=len(colMax), time_exe=time_instant,
                                 time_total=time_debut, g=G, cpt=0) + "\n"
        scoreMax, colMax = bestScore, bestCols
        # Main process iteration (generation iteration)
        while G < self.Gmax:
            instant = time.time()
            # Neighborhood exploration and evaluation
            neighborhood = self.create_population_models(inds=self.N, size=self.D + 1, models=self.model)
            # Evaluate the neighborhood
            scores, models, cols = fitness_models(train=self.train, test=self.test, pop=neighborhood,
                                                  target_name=self.target, metric=self.metric, model=self.model,
                                                  ratio=self.ratio)
            bestScore, worstScore, bestModel, bestInd, bestCols = \
                add(scores=scores, models=models, inds=np.asarray(neighborhood), cols=cols)
            bestInd = bestInd.tolist()
            G = G + 1
            same = same + 1
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            # Update which individual is the best
            if bestScore > scoreMax:
                same = 0
                scoreMax, modelMax, indMax, colMax = bestScore, bestModel, bestInd, bestCols
            print_out = self.sprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                     mean=mean_scores, worst=worstScore, feats=len(colMax), time_exe=time_instant,
                                     time_total=time_debut, g=G, cpt=same) + "\n"
            # If the time limit is exceeded, we stop
            if time.time() - debut >= self.Tmax:
                stop = True
            # Write important information to file
            if G % 10 == 0 or G == self.Gmax or stop:
                # Write important information to file
                self.specifics(colMax=colMax, bestScore=scoreMax, bestModel=modelMax, bestInd=indMax, g=G,
                               t=timedelta(seconds=(time.time() - debut)), last=G - same, out=print_out)
                print_out = ""
                if stop:
                    break
