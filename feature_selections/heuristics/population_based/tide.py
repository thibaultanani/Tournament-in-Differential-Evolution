import os
import random
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from skrebate import ReliefF
from utility.utility import createDirectory, add, get_entropy, create_population_models, fitness_ind_models,\
    fitness_models


class Tide(Heuristic):
    """
    Class that implements the new heuristic: tournament in differential evolution

    Args:
        alpha (float)     : The minimum threshold for the percentage of individuals to be selected by tournament
        k (int)           : Number of features to select when using filter method for initialisation
        filter_init (bool): The choice of using a filter method for initialisation
        entropy (float)   : Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, model, train, test=None, drops=None, metric=None, Tmax=None, ratio=None,
                 N=None, Gmax=None, alpha=None, k=None, filter_init=None, entropy=None, suffix=None):
        super().__init__(name, target, model, train, test, drops, metric, Tmax, ratio, suffix, N, Gmax)
        self.alpha = alpha or 0.9
        self.k = k or int(self.D / 2)
        if filter_init is False:
            self.filter_init = False
        else:
            self.filter_init = True
        self.entropy = entropy or 0.02
        self.path = os.path.join(self.path, 'tide' + self.suffix)
        createDirectory(path=self.path)

    def relief_init(self):
        X = self.train.drop([self.target], axis=1).values
        y = self.train[self.target].values
        relief = ReliefF(n_features_to_select=self.k)
        relief.fit(X, y)
        importances = relief.feature_importances_
        top_k_features = self.cols[importances.argsort()[-self.k:][::-1]]
        score, model, col, vector = 0, 0, 0, 0
        for i in range(len(self.model)):
            v = [0] * self.D
            for var in top_k_features:
                v[self.cols.get_loc(var)] = 1
            v.append(i)
            s, m, c = fitness_ind_models(train=self.train, test=self.test, ind=v, target_name=self.target,
                                         metric=self.metric, model=self.model[i], ratio=self.ratio)
            print("etape " + str(i) + ": ", s, len(c), self.model[v[-1]].__class__.__name__)
            if s > score:
                score, model, col, vector = s, m, c, v
        print("reliefF:", score, len(col), self.model[vector[-1]].__class__.__name__)
        return vector

    @staticmethod
    def mutate(P, n_ind, current, selected):
        list_of_index = [i for i in range(len(P))]
        r = np.random.choice(list_of_index, 2, replace=False)
        while current in r:
            r = np.random.choice(list_of_index, 2, replace=False)
        selected = [selected, r[0], r[1]]
        Xr1 = P[selected[0]]
        Xr2 = P[selected[1]]
        Xr3 = P[selected[2]]
        mutant = []
        for chromosome in range(n_ind):
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

    @staticmethod
    def tournament(scores, entropy, alpha):
        p = (1 - entropy) * (1 - alpha) + alpha
        nb_scores = int(scores.__len__() * p)
        selected = random.choices(scores, k=nb_scores)
        if len(selected) < 2:
            selected = random.choices(scores, k=2)
        score_max = np.amax(selected)
        for i, score in enumerate(scores):
            if score == score_max:
                return i

    def specifics(self, colMax, bestScore, bestModel, bestInd, g, t, last, out):
        if self.filter_init:
            string = "k: " + str(self.k)
            name = "Tournament In Differential Evolution + ReliefF"
        else:
            string = "k: No filter initialization"
            name = "Tournament In Differential Evolution"
        string = string + os.linesep + "Alpha: " + str(self.alpha) + os.linesep
        self.write(name, colMax, bestScore, bestModel, bestInd, g, t, last, string, out)

    def start(self, pid):
        code = "TIDE"
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
        r = None
        if self.filter_init:
            r = self.relief_init()
            P[0] = r
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
                # Calculate the rank for each individual
                selected = self.tournament(scores=scores, entropy=entropy, alpha=self.alpha)
                # Mutant calculation Vi
                Vi = self.mutate(P=P, n_ind=self.D + 1, current=i, selected=selected)
                # Child vector calculation Ui
                CR = random.uniform(0.3, 0.7)
                Ui = self.crossover(n_ind=self.D + 1, ind=P[i], mutant=Vi, cross_proba=CR)
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
                if self.filter_init:
                    P[0] = r
                scores, models, cols = \
                    fitness_models(train=self.train, test=self.test, pop=P, target_name=self.target,
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
