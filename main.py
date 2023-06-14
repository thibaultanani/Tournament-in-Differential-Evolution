from multiprocessing import Process

from feature_selections.filters import NoSelection, Correlation, Anova, MutualInformation, Mrmr, ReliefF
from feature_selections.wrappers import Rfe
from feature_selections.heuristics.population_based import Genetic, Differential, Pbil, Tide
from feature_selections.heuristics.single_solution import Tabu
from feature_selections.heuristics import Random

from utility import read, get_res
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    train = read(filename="madelon_train")
    test = read(filename="madelon_valid")
    name = "madelon"
    target = "Class"
    metric = "recall"
    tmax = 7200
    drops = []

    # For ALS only
    # drops = ['ID', 'Period', 'Source', 'Death Date']

    # For Madelon only
    tmp_train, tmp_test = train[target].values, test[target].values
    train = train[train.columns[:-1]].astype('float64')
    test = test[test.columns[:-1]].astype('float64')
    train[target], test[target] = tmp_train, tmp_test
    # -----------

    m = [RidgeClassifier(random_state=42), KNeighborsClassifier(), LinearSVC(random_state=42),
         RandomForestClassifier(random_state=42, n_estimators=10), LogisticRegression(random_state=42),
         LinearDiscriminantAnalysis(), GaussianNB()]

    rand = Random(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops)
    tabu = Tabu(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops)
    gene = Genetic(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops)
    diff = Differential(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax,
                        drops=drops)
    pbil = Pbil(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops)
    tide = Tide(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops,
                suffix='_reliefF', k=20)
    tide2 = Tide(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops,
                 filter_init=False)
    all_ = NoSelection(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops)
    corr = Correlation(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops,
                       k=20)
    anov = Anova(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops, k=20)
    info = MutualInformation(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax,
                             drops=drops, k=20)
    mrmr = Mrmr(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops,
                k=20)
    reli = ReliefF(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops,
                   k=20)
    rfe_ = Rfe(name=name, target=target, train=train, test=test, metric=metric, model=m, Tmax=tmax, drops=drops, k=20)

    methods = [rand, tabu, gene, diff, pbil, tide, tide2, all_, corr, anov, info, mrmr, reli, rfe_]

    processes = []
    for i in range(len(methods)):
        if type(methods[i]) in [type(x) for x in methods[:i]]:
            processes.append(Process(target=methods[i].start, args=(2,)))
        else:
            processes.append(Process(target=methods[i].start, args=(1,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    get_res(foldername=name)

    print("Finish !")
