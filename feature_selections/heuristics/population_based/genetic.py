import os
import random
import time
import numpy as np

from feature_selections.heuristics.heuristic import Heuristic
from datetime import timedelta
from utility import createDirectory, add, get_entropy, create_population_models, fitness_models, fitness_ind_models


class Genetic(Heuristic):
    """
    Class that implements the genetic algorithm heuristic.

    Args:
        mutation (float): Maximum number of mutations for each child
        elite (float)   : Number of individuals to keep before creating children
        entropy (float) : Minimum threshold of diversity in the population to be reached before a reset
    """
    def __init__(self, name, target, train, test, model, drops=None, metric=None, Tmax=None, ratio=None, N=None,
                 Gmax=None, mutation=None, elite=None, entropy=None, suffix=None):
        super().__init__(name, target, model, train, test, drops, metric, Tmax, ratio, suffix, N, Gmax)
        self.mutation = mutation or 5
        self.elite = elite or int(self.N / 2)
        self.entropy = entropy or 0.02
        self.path = os.path.join(self.path, 'genetic' + self.suffix)
        createDirectory(path=self.path)

    @staticmethod
    def get_ranks(scores):
        ranks = {}
        for i, score in enumerate(sorted(scores)):
            rank = i + 1
            if score not in ranks:
                ranks[score] = rank
        return [ranks[score] for score in scores]

    @staticmethod
    def mutate(individual, mutations, models):
        mutant = individual.copy()
        bits_to_flip = random.sample(range(len(individual)), random.randint(0, mutations))
        for chromosome in bits_to_flip:
            if chromosome != len(individual) - 1:
                mutant[chromosome] = int(not mutant[chromosome])
            else:
                r = random.randint(0, len(models) - 1)
                while r == mutant[chromosome]:
                    r = random.randint(0, len(models) - 1)
                mutant[chromosome] = r
        return mutant

    @staticmethod
    def crossover(parent1, parent2):
        parent1 = parent1.tolist()
        parent2 = parent2.tolist()
        if random.random() < 0.5:
            p1, p2 = parent2, parent1
        else:
            p1, p2 = parent1, parent2
        point1 = random.randint(0, len(p1) - 1)
        point2 = random.randint(point1, len(p1) - 1)
        child = p1[:point1] + p2[point1:point2] + p1[point2:]
        return child

    @staticmethod
    def print_(print_out, pid, maxi, best, mean, worst, feats, time_exe, time_total, entropy, g, cpt):
        display = "[GENE]    PID: [{:3}]    G: {:5d}    max: {:2.4%}    features: {:6d}    best: {:2.4%}" \
                  "   mean: {:2.4%}    worst: {:2.4%}    G time: {}    T time: {}    last: {:6d}    entropy : {:2.3%}" \
            .format(pid, g, maxi, feats, best, mean, worst, time_exe, time_total, cpt, entropy)
        print_out = print_out + display
        print(display)
        return print_out

    def specifics(self, colMax, bestScore, bestModel, bestInd, g, t, last, out):
        string = "Mutation Rate: " + str(self.mutation) + os.linesep
        self.write("Genetic Algorithm", colMax, bestScore, bestModel, bestInd, g, t, last, string, out)

    def start(self, pid, result_queue):
        code = "GENE"
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
            # Keeping only elite size of the population for the next generation
            indexes = np.argsort(scores)[::-1][:self.elite]
            scores = [scores[i] for i in indexes]
            models = [models[i] for i in indexes]
            cols = [cols[i] for i in indexes]
            P = [P[i] for i in indexes]
            # Calculate the rank for each individual
            ranks = self.get_ranks(scores=scores)
            probas = [n / sum(ranks) for n in ranks]
            list_of_index = [i for i in range(0, len(probas))]
            # Children population
            for i in range(self.N):
                parents = np.random.choice(list_of_index, 2, p=probas, replace=False)
                child = self.crossover(parent1=P[parents[0]], parent2=P[parents[1]])
                child = self.mutate(individual=child, mutations=self.mutation, models=self.model)
                score_, model_, col_ = \
                    fitness_ind_models(train=self.train, test=self.test, ind=child, target_name=self.target,
                                       metric=self.metric, model=self.model[child[-1]], ratio=self.ratio)
                scores.append(score_)
                models.append(model_)
                cols.append(col_)
                P.append(np.asarray(child))
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
