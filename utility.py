import math
import os
import random
import shutil
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


def read(filename, separator=','):
    path = os.path.join(os.getcwd(), os.path.join('datasets', filename))
    try:
        data = pd.read_excel(path + '.xlsx', index_col=None, engine='openpyxl')
    except FileNotFoundError:
        data = pd.read_csv(path + '.csv', index_col=None, sep=separator)
    return data


def write(filename, data):
    path = os.path.join(os.getcwd(), os.path.join('datasets', filename))
    try:
        data.to_excel(path + '.xlsx', index=False)
    except FileNotFoundError:
        data.to_csv(path + '.csv', index=False)
    return data


def createDirectory(path):
    # Clears the record of previous experiments on the same dataset with the same heuristic
    final = path
    if os.path.exists(final):
        shutil.rmtree(final)
    os.makedirs(final)


def create_population(inds, size):
    # Initialise the population
    pop = np.random.rand(inds, size) < np.random.rand(inds, 1)
    pop = pop[:, np.argsort(-np.random.rand(size), axis=0)]
    return pop.astype(bool)


def create_population_models(inds, size, models):
    # Initialise the population
    pop = np.random.rand(inds, size) < np.random.rand(inds, 1)
    pop = pop[:, np.argsort(-np.random.rand(size), axis=0)]
    pop = pop.astype(int)
    # Replace last element with random integer between 0 and models-1
    pop[:, -1] = np.random.randint(0, len(models), size=inds)
    return pop


def preparation(train, test, ind, target):
    # Selects columns based on the value of an individual
    copy = train.copy()
    copy_target = copy[target]
    copy = copy.drop([target], axis=1)
    cols = copy.columns
    cols_selection = []
    for c in range(len(cols)):
        if ind[c]:
            cols_selection.append(cols[c])
    copy = copy[cols_selection]
    copy[target] = copy_target
    if isinstance(test, pd.DataFrame):
        copy2 = test.copy()
        copy_target2 = copy2[target]
        copy2 = copy2.drop([target], axis=1)
        copy2 = copy2[cols_selection]
        copy2[target] = copy_target2
        return copy, copy2, cols_selection
    else:
        return copy, None, cols_selection


def learning(train, test, target, model):
    # Performs learning according to the chosen method
    X, y = train.drop(target, axis=1).values, train[target].values
    if isinstance(test, pd.DataFrame):
        X_train, y_train = X, y
        X_test, y_test = test.drop(target, axis=1).values, test[target].values
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)
    else:
        matrix_length = len(np.unique(train[target]))
        matrix = np.zeros((matrix_length, matrix_length), dtype=int)
        originalclass = []
        predictedclass = []
        k = StratifiedKFold(n_splits=5)
        for train_index, test_index in k.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            originalclass.extend(y_test)
            predictedclass.extend(y_pred)
            matrix = matrix + confusion_matrix(y_test, y_pred)
        y_test, y_pred = originalclass, predictedclass
    return accuracy_score(y_true=y_test, y_pred=y_pred), \
           precision_score(y_true=y_test, y_pred=y_pred, average="macro"), \
           recall_score(y_true=y_test, y_pred=y_pred, average="macro"), matrix


def fitness_ind(train, test, ind, target_name, metric, model, ratio):
    # Process of calculating the fitness for a single individual/subset
    if not any(ind):
        ind[random.randint(0, len(ind) - 1)] = True
    train_, test_, cols = preparation(train=train, test=test, ind=ind, target=target_name)
    accuracy, precision, recall, matrix = learning(train=train_, test=test_, target=target_name, model=model)
    metric = metric.lower()
    if metric == 'accuracy':
        score = accuracy
    elif metric == 'recall':
        score = recall
    elif metric == 'precision':
        score = precision
    else:
        score = accuracy
    score = score - (ratio * (len(cols) / train.shape[1]))
    return score, matrix, cols


def fitness(train, test, pop, target_name, metric, model, ratio):
    # Process of calculating the fitness for each individual of a population
    score_list, accuracy_list, precision_list, recall_list, fscore_list, col_list, matrix_list = \
        [], [], [], [], [], [], []
    metric = metric.lower()
    for ind in pop:
        if not any(ind):
            ind[random.randint(0, len(ind) - 1)] = True
        train_, test_, cols = preparation(train=train, test=test, ind=ind, target=target_name)
        accuracy, precision, recall, matrix = learning(train=train_, test=test_, target=target_name, model=model)
        if metric == 'accuracy':
            score = accuracy
        elif metric == 'recall':
            score = recall
        elif metric == 'precision':
            score = precision
        else:
            score = accuracy
        score_list.append(score - (ratio * (len(cols) / train.shape[1])))
        col_list.append(cols)
        matrix_list.append(matrix)
    return score_list, matrix_list, col_list


def fitness_ind_models(train, test, ind, target_name, metric, model, ratio):
    # Process of calculating the fitness for a single individual/subset
    metric = metric.lower()
    if not any(ind[:-1]):
        ind[random.randint(0, len(ind) - 2)] = 1
    train_, test_, cols = preparation(train=train, test=test, ind=ind, target=target_name)
    accuracy, precision, recall, matrix = learning(train=train_, test=test_, target=target_name, model=model)
    if metric == 'accuracy':
        score = accuracy
    elif metric == 'recall':
        score = recall
    elif metric == 'precision':
        score = precision
    else:
        score = accuracy
    score = score - (ratio * (len(cols) / train.shape[1]))
    return score, matrix, cols


def fitness_models(train, test, pop, target_name, metric, model, ratio):
    # Process of calculating the fitness for each individual of a population
    score_list, accuracy_list, precision_list, recall_list, fscore_list, col_list, matrix_list = \
        [], [], [], [], [], [], []
    metric = metric.lower()
    for ind in pop:
        if not any(ind[:-1]):
            ind[random.randint(0, len(ind) - 2)] = 1
        train_, test_, cols = preparation(train=train, test=test, ind=ind, target=target_name)
        accuracy, precision, recall, matrix =\
            learning(train=train_, test=test_, target=target_name, model=model[ind[-1]])
        if metric == 'accuracy':
            score = accuracy
        elif metric == 'recall':
            score = recall
        elif metric == 'precision':
            score = precision
        else:
            score = accuracy
        score_list.append(score - (ratio * (len(cols) / train.shape[1])))
        col_list.append(cols)
        matrix_list.append(matrix)
    return score_list, matrix_list, col_list


def get_entropy(pop):
    H = []
    # Loop over the columns
    for i in range(len(pop[0])):
        # Initialize variables to store the counts of True and False values
        true_count = 0
        false_count = 0
        # Loop over the rows and count the number of True and False values in the current column
        for row in pop:
            if row[i]:
                true_count += 1
            else:
                false_count += 1
        # Calculate the probabilities of True and False values
        p_true = true_count / (true_count + false_count)
        p_false = false_count / (true_count + false_count)
        # Calculate the Shannon's entropy for the current column
        if p_true == 0 or p_false == 0:
            entropy = 0
        else:
            entropy = -(p_true * math.log2(p_true) + p_false * math.log2(p_false))
        # Append the result to the list
        H.append(entropy)
    return sum(H) / len(H)


def add(scores, models, inds, cols):
    argmax = np.argmax(scores)
    argmin = np.argmin(scores)
    bestScore = scores[argmax]
    worstScore = scores[argmin]
    bestModel = models[argmax]
    bestInd = inds[argmax]
    bestCols = cols[argmax]
    return bestScore, worstScore, bestModel, bestInd, bestCols


def get_res(foldername):
    folders = []
    file_to_search = os.path.join("out", foldername)
    for filename in os.listdir(file_to_search):
        if os.path.isdir(os.path.join(file_to_search, filename)):
            folders.append(filename)
    heuristics, types, methods, scores, balanced, features, iters, max_iters, times, ranks =\
        [], [], [], [], [], [], [], [], [], []
    for folder in folders:
        if folder != 'filters':
            f = open(os.path.join(os.path.join("out", foldername), os.path.join(folder, "results.txt")), 'r')
            recall = 0
            gen = False
            m = None
            div = 0
            lines = f.readlines()
            for line in lines:
                if line.startswith("Metaheuristic: "):
                    heuristics.append(line.split(": ")[1].split("\n")[0])
                    types.append("Metaheuristic")
                if line.startswith("Filter: "):
                    heuristics.append(line.split(": ")[1].split("\n")[0])
                    types.append("Filter")
                if line.startswith("Wrapper: "):
                    heuristics.append(line.split(": ")[1].split("\n")[0])
                    types.append("Wrapper")
                if line.startswith("Best Method: "):
                    methods.append(line.split(": ")[1].split("\n")[0])
                if line.startswith("Best Score: "):
                    scores.append('{:.4%}'.format(eval(line.split(": ")[1].split("\n")[0])))
                if line.startswith("Class"):
                    values = line.split()
                    TP = int(values[3])
                    # TN = int(values[5])
                    # FP = int(values[7])
                    # FN = int(values[9])
                    Total = int(values[11])
                    recall = recall + (TP / Total)
                    div = div + 1
                if line.startswith("Number of Features: "):
                    features.append(line.split(": ")[1].split("\n")[0])
                if line.startswith("Generation Performed: "):
                    iters.append(eval(line.split(": ")[1].split("\n")[0]))
                    gen = True
                if line.startswith("Latest Improvement: "):
                    m = eval(line.split(": ")[1].split("\n")[0])
                if line.startswith("Execution Time: "):
                    duration = line.split(": ")[1].split(" ")[0]
                    times.append(duration)
            if not gen:
                iters.append(None)
                max_iters.append(None)
            else:
                max_iters.append(m)
            f.close()
            balanced_accuracy = recall / div
            # print(folder, balanced_accuracy, recall, div)
            balanced.append('{:.4%}'.format(balanced_accuracy))
    order = [2, 1, 11, 9, 4, 3, 0, 10, 7, 5, 6, 8, 12, 13]
    heuristics = [x for _, x in sorted(zip(order, heuristics))]
    types = [x for _, x in sorted(zip(order, types))]
    methods = [x for _, x in sorted(zip(order, methods))]
    scores = [x for _, x in sorted(zip(order, scores))]
    balanced = [x for _, x in sorted(zip(order, balanced))]
    features = [x for _, x in sorted(zip(order, features))]
    iters = [x for _, x in sorted(zip(order, iters))]
    max_iters = [x for _, x in sorted(zip(order, max_iters))]
    times = [x for _, x in sorted(zip(order, times))]
    # Convertir les scores en nombres décimaux
    scores_ = [float(score.strip('%')) for score in scores]
    # Trier les scores par ordre décroissant
    sorted_scores = sorted(scores_, reverse=True)
    # Créer la liste des rangs
    ranks = [sorted_scores.index(score) + 1 for score in scores_]
    data = pd.DataFrame()
    data["Selection Method"] = heuristics
    data["Type"] = types
    data["Learning Method (Max Score)"] = methods
    data["Score"] = [s.replace('.', ',') for s in scores]
    data["Balanced Accuracy"] = [b.replace('.', ',') for b in balanced]
    data["Features"] = features
    data["Iterations"] = iters
    data["Iterations (Max Score)"] = max_iters
    data["Time"] = times
    data["Rank"] = ranks
    data.to_excel(os.path.join(os.path.join("out", foldername), "summary.xlsx"), index=False)

