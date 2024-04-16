import numpy as np
from skmultilearn.dataset import load_from_arff
from skmultilearn.model_selection import IterativeStratification

def read_arff(path, label_count, wantfeature=False):
    path_to_arff_file=path+".arff"
    arff_file_is_sparse = False
    X, Y, feature_names, label_names = load_from_arff(
        path_to_arff_file,
        label_count=label_count,
        label_location="end",
        load_sparse=arff_file_is_sparse,
        return_attribute_definitions=True
    )
    if(~wantfeature):
        return X, Y, None
    else:
        featype = []
        for i in range(len(feature_names)):
            if(feature_names[i][1] == 'NUMERIC'):
                featype.append([0])
            else:
                if(not feature_names[i][1][0].isdigit()):
                    feature_nomimal = np.arange(0,len(feature_names[i][1]))
                    featype.append([int(number) for number in feature_nomimal])
                else:
                    featype.append([int(number) for number in feature_names[i][1]])
        return X, Y, featype

class ReadData:
    def __init__(self, genpath="data/"):
        self.genpath = genpath
        '''ALL datasets from KDIS (http://www.uco.es/kdis/mllresources/)'''
        self.datasnames = ["CHD_49","Emotions","Foodtruck","GnegativeGO","HumanGO","Image","Langlog","Scene","Chess","Tmc2007_500",
            "Water_quality","Business","Entertainment","Yeast","Yelp"]
        self.dimALL = [555,593,407,1392,3106,2000,1460,2407,1675,28600,1060,11210,12730,2417,10810]
        self.num_labels = [6,6,12,8,14,5,75,6,227,22,14,30,21,14,5]
        self.dimTrains = [372,395,275,931,2053,1501,978,1618,1107,19140,710,7523,8569,1629,7240]
        self.dimTests = [183,198,132,461,1053,499,482,789,568,9456,350,3691,4161,788,3566]

    def readData_org(self, index):
        label_count = self.num_labels[index]
        path = self.genpath + self.datasnames[index]
        X, Y, featype = read_arff(path, label_count, False)
        dimTrain = self.dimTrains[index]
        dimTest = self.dimTests[index]
        print(self.datasnames[index],np.shape(X),np.shape(Y),dimTrain,dimTest)
        train_idx = np.arange(dimTrain)
        test_idx = np.arange(dimTrain,dimTrain+dimTest)
        return X[train_idx],Y[train_idx],X[test_idx],Y[test_idx],featype

    def readData(self, index):
        X,Y,Xt,Yt,f = self.readData_org(index)
        X,Y,Xt,Yt = np.array(X.todense()), np.array(Y.todense()), np.array(Xt.todense()), np.array(Yt.todense())
        return X,Y,Xt,Yt

    def readData_CV(self, index, CV=10):
        label_count = self.num_labels[index]
        # print(self.datasnames[index],self.dimALL[index],self.num_labels[index])
        path = self.genpath + self.datasnames[index]
        X, Y, f = read_arff(path, label_count)
        k_fold = IterativeStratification(n_splits=CV, order=1)
        # for train, test in k_fold.split(X, Y):
        #     print(np.shape(train),np.shape(test))
        return k_fold, np.array(X.todense()), np.array(Y.todense())
    
    def getnum_label(self):
        return self.num_labels

def resolveResult(dataName='', algName='', result=[], time1=0, time2=0, time3=0):
    f = open('result.txt', 'a')
    print(dataName, end='\t', file=f)
    print(algName, end='\t', file=f)
    for i in range(np.shape(result)[0]):
        print(result[i], end='\t', file=f)
    if(time1>0):
        print(time1, end='\t', file=f)
    if(time2>0):
        print(time2, end='\t', file=f)
    if(time3>0):
        print(time3, end='\t', file=f)
    print('', file=f)
    f.close()
