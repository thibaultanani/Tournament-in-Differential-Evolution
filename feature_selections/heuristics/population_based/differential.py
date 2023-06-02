import os
import random
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility import createDirectory, add, get_entropy, create_population_models, fitness_ind_models, fitness_models


class Differential(Heuristic):
    """
    Class that implements the differential evolution heuristic.

    Args:
        F (float)      : Probability factor controlling the amplification of the differential variation
        CR (float)     : Crossover probability
        entropy (float): Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, model, train, test=None, drops=None, metric=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, F=None, CR=None, entropy=None, suffix=None):
        super().__init__(name, target, model, train, test, drops, metric, Tmax, ratio, suffix, N, Gmax)
        self.F = F or 1.0
        self.CR = CR or 0.5
        self.entropy = entropy or 0.02
        self.path = os.path.join(self.path, 'differential' + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def mutate(P, n_ind, F, current, best):
        list_of_index = [i for i in range(len(P))]
        selected = np.random.choice(list_of_index, 3, replace=False)
        while (current or best) in selected:
            selected = np.random.choice(list_of_index, 3, replace=False)
        Xr1 = [int(x) for x in P[best]]
        Xr2 = [int(x) for x in P[selected[0]]]
        Xr3 = [int(x) for x in P[selected[1]]]
        mutant = []
        for chromosome in range(n_ind):
            if chromosome != n_ind - 1:
                val = Xr1[chromosome] + F * (Xr2[chromosome] - Xr3[chromosome])
                rounded_num = round(val, 1)
                result = max(0, min(1, rounded_num))
                if result == 0:
                    mutant.append(0)
                else:
                    mutant.append(1)
            else:
                if Xr2[chromosome] == Xr3[chromosome]:
                    mutant.append(Xr1[chromosome])
                else:
                    mutant.append(Xr2[chromosome])
        return mutant

    @staticmethod
    def crossover(n_ind, ind, mutant, cross_proba):
        cross_points = np.random.rand(n_ind) <= cross_proba
        child = np.where(cross_points, mutant, ind)
        jrand = random.randint(0, n_ind - 1)
        child[jrand] = mutant[jrand]
        return child

    def specifics(self, colMax, bestScore, bestModel, bestInd, g, t, last, out):
        string = "F factor: " + str(self.F) + os.linesep + "Crossover rate: " + str(self.CR) + os.linesep
        self.write("Differential Evolution", colMax, bestScore, bestModel, bestInd, g, t, last, string, out)

    def start(self, pid, result_queue):
        code = "DIFF"
        debut = time.time()
        old_path = self.path
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        # Measuring the execution time
        instant = time.time()
        # Generation (G) initialisation
        G, same1, same2, stop = 0, 0, 0, False
        # Population P initialisation
        P = create_population_models(inds=self.N, size=self.D + 1, models=self.model)
        # Evaluates population
        scores, models, cols = fitness_models(train=self.train, test=self.test, pop=P, target_name=self.target,
                                              metric=self.metric, model=self.model, ratio=self.ratio)
        bestScore, worstScore, bestModel, bestInd, bestCols = \
            add(scores=scores, models=models, inds=np.asarray(P), cols=cols)
        scoreMax, modelMax, indMax, colMax = bestScore, bestModel, bestInd, bestCols
        mean_scores = float(np.mean(scores))
        time_instant = timedelta(seconds=(time.time() - instant))
        time_debut = timedelta(seconds=(time.time() - debut))
        # Calculate diversity in population
        entropy = get_entropy(pop=P)
        # Pretty print the results
        print_out = self.pprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                 mean=mean_scores, worst=worstScore, feats=len(colMax), time_exe=time_instant,
                                 time_total=time_debut, entropy=entropy, g=G, cpt=0) + "\n"
        # Main process iteration (generation iteration)
        while G < self.Gmax:
            instant = time.time()
            # Mutant population creation and evaluation
            for i in range(self.N):
                # Mutant calculation Vi
                Vi = self.mutate(P=P, n_ind=self.D + 1, F=self.F, current=i, best=np.argmax(scores))
                # Child vector calculation Ui
                Ui = self.crossover(n_ind=self.D + 1, ind=P[i], mutant=Vi, cross_proba=self.CR)
                # Evaluation of the trial vector
                if all(x == y for x, y in zip(P[i], Ui)):
                    score_, model_, col_ = scores[i], models[i], cols[i]
                else:
                    score_, model_, col_ = \
                        fitness_ind_models(train=self.train, test=self.test, ind=Ui, target_name=self.target,
                                           metric=self.metric, model=self.model[Ui[-1]], ratio=self.ratio)
                # Comparison between Xi and Ui
                if scores[i] <= score_:
                    # Update population
                    P[i], scores[i], models[i], cols[i] = Ui, score_, model_, col_
                    bestScore, worstScore, bestModel, bestInd, bestCols = \
                        add(scores=scores, models=models, inds=np.asarray(P), cols=cols)
            G = G + 1
            same1, same2 = same1 + 1, same2 + 1
            mean_scores = float(np.mean(scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))
            entropy = get_entropy(pop=P)
            # Update which individual is the best
            if bestScore > scoreMax:
                same1, same2 = 0, 0
                scoreMax, modelMax, indMax, colMax = bestScore, bestModel, bestInd, bestCols
            print_out = self.pprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                     mean=mean_scores, worst=worstScore, feats=len(colMax), time_exe=time_instant,
                                     time_total=time_debut, entropy=entropy, g=G, cpt=same2) + "\n"
            # If diversity is too low restart
            if entropy < self.entropy or same1 >= 300:
                same1 = 0
                P = create_population_models(inds=self.N, size=self.D + 1, models=self.model)
                # P[0] = indMax
                scores, models, cols = fitness_models(train=self.train, test=self.test, pop=P, target_name=self.target,
                                                      metric=self.metric, model=self.model, ratio=self.ratio)
                bestScore, worstScore, bestModel, bestInd, bestCols = \
                    add(scores=scores, models=models, inds=np.asarray(P), cols=cols)
            # If the time limit is exceeded, we stop
            if time.time() - debut >= self.Tmax:
                stop = True
            # Write important information to file
            if G % 10 == 0 or G == self.Gmax or stop:
                self.specifics(colMax=colMax, bestScore=scoreMax, bestModel=modelMax, bestInd=indMax, g=G,
                               t=timedelta(seconds=(time.time() - debut)), last=G - same2, out=print_out)
                print_out = ""
                if stop:
                    break
        result_queue.put((pid, scoreMax, time_debut, G - same2, len(colMax), old_path))
