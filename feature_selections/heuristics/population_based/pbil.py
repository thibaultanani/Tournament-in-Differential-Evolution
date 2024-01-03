import os
import random
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from copy import copy
from utility.utility import createDirectory, add, get_entropy, fitness_models


class Pbil(Heuristic):
    """
    Class that implements the population based incremental learning heuristic.

    Args:
        LR (float)      : Learning rate, the speed at which the proba vector will converge
        MP (float)      : Mutation propability on a value in the proba vector
        MS (float)      : Mutation shift, the speed at which the proba vector will converge when a mutation occurs
        entropy (float) : Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, train, test, model, drops=None, metric=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, LR=None, MP=None, MS=None, entropy=None, suffix=None):
        super().__init__(name, target, model, train, test, drops, metric, Tmax, ratio, suffix, N, Gmax)
        self.LR = LR or 0.1
        self.MP = MP or 0.05
        self.MS = MS or 0.1
        self.entropy = entropy or 0.02
        self.path = os.path.join(self.path, 'pbil' + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def create_probas(size):
        probas = []
        for i in range(size):
            probas.append(0.5)
        return probas

    @staticmethod
    def create_probas_models(size):
        probas = []
        for i in range(size):
            probas.append(1 / size)
        return probas

    @staticmethod
    def update_probas(probas, LR, best):
        for i in range(len(probas)):
            probas[i] = (probas[i] * (1.0 - LR)) + (LR * best[i])
        return probas

    @staticmethod
    def update_probas_models(m_probas, LR, best):
        for i in range(len(m_probas)):
            if best[-1] == i:
                m_probas[i] = (m_probas[i] * (1.0 - LR)) + (LR * 1)
            else:
                m_probas[i] = (m_probas[i] * (1.0 - LR)) + (LR * 0)
        return m_probas

    @staticmethod
    def mutate_probas(probas, MP, MS):
        for i in range(len(probas)):
            if random.random() < MP:
                probas[i] = (probas[i] * (1.0 - MS)) + (MS * random.choice([1.0, 0.0]))
        return probas

    @staticmethod
    def mutate_probas_models(m_probas, MP, MS):
        for i in range(len(m_probas)):
            if random.random() < MP:
                r = random.choice([1, 0])
                for j in range(len(m_probas)):
                    if j == i:
                        m_probas[j] = (m_probas[j] * (1.0 - MS)) + (MS * r)
                    else:
                        m_probas[j] = (m_probas[j] * (1.0 - MS)) + (MS * (1 - r))
        return [p / sum(m_probas) for p in m_probas]

    @staticmethod
    def create_population(inds, size, probas, m_probas):
        pop = np.zeros((inds, size), dtype=int)
        for i in range(inds):
            for j in range(len(pop[i])):
                if j != len(pop[i]) - 1:
                    if random.random() < probas[j]:
                        pop[i][j] = 1
                    else:
                        pop[i][j] = 0
                else:
                    pop[i][j] = random.choices(range(len(m_probas)), m_probas)[0]
        return pop

    def specifics(self, probas, m_probas, colMax, bestScore, bestModel, bestInd, g, t, last, out):
        string = "Learning Rate: " + str(self.LR) + os.linesep + \
                 "Mutation Probabilities: " + str(self.MP) + os.linesep + \
                 "Mutation Shift: " + str(self.MS) + os.linesep + \
                 "Probabilities Vector: " + str(['%.3f' % x for x in probas]) + os.linesep + \
                 "Probabilities Vector Models: " + str(['%.3f' % x for x in m_probas]) + os.linesep
        self.write("Population Based Incremental Learning", colMax, bestScore, bestModel, bestInd, g, t, last, string,
                   out)

    def start(self, pid):
        code = "PBIL"
        debut = time.time()
        old_path = self.path
        self.path = os.path.join(self.path)
        createDirectory(path=self.path)
        print_out = ""
        np.random.seed(None)
        scoreMax, modelMax, indMax, colMax = 0, 0, 0, 0
        # Generation (G) initialisation
        G, same1, same2, time_debut, stop = 0, 0, 0, 0, False
        # Initialize probabilities vector (0.5 to select each features)
        probas = self.create_probas(size=self.D)
        m_probas = self.create_probas_models(size=len(self.model))
        saves_proba, saves_proba_m = copy(probas), copy(m_probas)
        # Main process iteration (generation iteration)
        while G < self.Gmax:
            # Measuring the execution time
            instant = time.time()
            P = self.create_population(inds=self.N, size=self.D, probas=probas, m_probas=m_probas)
            # Evaluates population
            scores, models, cols = fitness_models(train=self.train, test=self.test, pop=P, target_name=self.target,
                                                  metric=self.metric, model=self.model, ratio=self.ratio)
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
                saves_proba, saves_proba_m = copy(probas), copy(m_probas)
            print_out = self.pprint_(print_out=print_out, name=code, pid=pid, maxi=scoreMax, best=bestScore,
                                     mean=mean_scores, worst=worstScore, feats=len(colMax), time_exe=time_instant,
                                     time_total=time_debut, entropy=entropy, g=G, cpt=same2) + "\n"
            # If the time limit is exceeded, we stop
            if time.time() - debut >= self.Tmax:
                stop = True
            # Write important information to file
            if G % 10 == 0 or G == self.Gmax or stop:
                self.specifics(probas=saves_proba, m_probas=saves_proba_m, colMax=colMax, bestScore=scoreMax,
                               bestModel=modelMax, bestInd=indMax, g=G, t=timedelta(seconds=(time.time() - debut)),
                               last=G - same2, out=print_out)
                print_out = ""
                if stop:
                    break
            # Update probabilies vectors with the best individual of the generation
            probas = self.update_probas(probas=probas, LR=self.LR, best=bestInd)
            m_probas = self.update_probas_models(m_probas=m_probas, LR=self.LR, best=bestInd)
            # Mutations to increase diversity and premature convergence
            probas = self.mutate_probas(probas=probas, MP=self.MP, MS=self.MS)
            m_probas = self.mutate_probas_models(m_probas=m_probas, MP=self.MP, MS=self.MS)
            # If diversity is too low restart
            if entropy < self.entropy or same1 >= 300:
                same1 = 0
                probas = self.create_probas(size=self.D)
                m_probas = self.create_probas_models(size=len(self.model))
