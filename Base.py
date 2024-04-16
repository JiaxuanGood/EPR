from time import time
import random
import numpy as np
from mReadData import *
from skmultilearn.problem_transform.br import BinaryRelevance
from skmultilearn.problem_transform.cc import ClassifierChain
from skmultilearn.problem_transform.lp import LabelPowerset

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.adapt import MLkNN

from mEvaluation import evaluate

def base_cls(mod='svm2'):
    if(mod=='svm'):
        return SVC(probability=True, tol=1e-4, cache_size=200, max_iter=-1)
    elif(mod=='sgd'):
        return SGDClassifier()
    elif(mod=='svm2'):
        return SVC(probability=True, tol=0.1, cache_size=200, max_iter=1000)
    elif(mod=='bayes'):
        return GaussianNB()
    elif(mod=='dt'):
        return DecisionTreeClassifier()
    elif(mod=='nn'):
        return MLPClassifier(tol=1e-4, max_iter=200)
    elif(mod=='forest'):
        return RandomForestClassifier()
    else:
        return None

# def fill1(Y):
#     Y = np.array(Y)
#     for j in range(np.shape(Y)[1]):
#         if(np.sum(Y[:,j])==0):
#             Y[0][j] = 1
#     return Y
