from Base import *
from OutLearner import Out
from missingData import *
from CorrGraph import *
from mReadData import *

'''additional BR classifier'''
def basing(X,Y,Xt):
    br = BinaryRelevance(classifier=base_cls())
    br.fit(X,fill1(intlabel(Y)))
    prediction = br.predict_proba(Xt)
    return prediction

'''semi-completed label -> ranking classification label'''
def rankLabel(y1,y2):
    y_equa,y_rank = [],[]
    idx_equa,idx_rank = [],[]
    y = []
    y_weight = []
    for i in range(len(y1)):
        if(y1[i]==y2[i]):
            idx_equa.append(i)
            if(y1[i]>=0.5):
                y_equa.append(1)
                y.append(1)
            else:
                y_equa.append(0)
                y.append(0)
        else:
            this_weight = np.abs(y1[i]-y2[i])
            y_weight.append(this_weight)
            idx_rank.append(i)
            if(y1[i]>y2[i]):
                y_rank.append(1)
                y.append(1)
            else:
                y_rank.append(0)
                y.append(0)
    # winners = np.argwhere(y_weight > quantile(y_weight))
    # winners = winners.flatten()
    # y_rank = np.array(y_rank).take(winners)
    # idx_rank = np.array(idx_rank).take(winners)
    # y_weight = np.array(y_weight).take(winners)
    return y_equa,y_rank,idx_equa,idx_rank,y_weight

def induce_aLearner(X,y,y_weight):
    if(np.sum(y)==0):
        print(0)
        return Out(0)
    if(np.sum(y)==len(y)):
        print(1)
        return Out(1)
    cls = base_cls()
    y_weight = np.array(y_weight)
    y_weight = y_weight/np.sum(y_weight) * len(y_weight)
    cls.fit(X,y,y_weight)
    return cls

'''full connected graph'''
def RANKf(X,Y_new,Xt,Yt,basis=np.zeros((1,1))):
    M,Q = np.shape(Yt)
    prediction = np.zeros((Q,M))
    t1,t2 = 0,0
    for i in range(Q):
        for j in range(i+1,Q):
            t0 = time()
            y_equa,y_rank,idx_equa,idx_rank,y_weight = rankLabel(Y_new[:,i],Y_new[:,j])
            thisLearner = induce_aLearner(X[idx_rank], y_rank, y_weight)
            t1 += time()-t0
            t0 = time()
            prd_rank = thisLearner.predict_proba(Xt)
            prd_rank = np.transpose(prd_rank)
            alpha = 2*np.abs(prd_rank[1]-0.5)
            if(np.shape(basis)==(1,1)):
                prediction[i] += prd_rank[1]
                prediction[j] += prd_rank[0]
            else:
                prediction[i] += alpha*prd_rank[1]+(1-alpha)*basis[:,i]
                prediction[j] += alpha*prd_rank[0]+(1-alpha)*basis[:,j]
            t2 += time()-t0
    prediction /= (Q-1)
    prediction = np.transpose(prediction)
    return prediction,t1,t2

'''a heuristic ranking system'''
def RANKa(X,Y_new,Xt,Yt,elist,basis=np.zeros((1,1))):
    M,Q = np.shape(Yt)
    cnts = np.zeros(Q)
    tmp = np.array(elist).flatten().tolist()
    for i in range(len(tmp)):
        cnts[tmp[i]] += 1
    prediction = np.zeros((Q,M))
    t1,t2 = 0,0
    for r in range(Q-1):
        i=elist[r][0]
        j=elist[r][1]
        t0 = time()
        y_equa,y_rank,idx_equa,idx_rank,y_weight = rankLabel(Y_new[:,i],Y_new[:,j])
        thisLearner = induce_aLearner(X[idx_rank], y_rank, y_weight)
        t1 += time()-t0
        t0 = time()
        prd_rank = thisLearner.predict_proba(Xt)
        prd_rank = np.transpose(prd_rank)
        alpha = 2*np.abs(prd_rank[1]-0.5)
        if(np.shape(basis)==(1,1)):
            prediction[i] += prd_rank[1]
            prediction[j] += prd_rank[0]
        else:
            prediction[i] += alpha*prd_rank[1]+(1-alpha)*basis[:,i]
            prediction[j] += alpha*prd_rank[0]+(1-alpha)*basis[:,j]
        t2 += time()-t0
    for i in range(Q):
        prediction[i] /= cnts[i]
    prediction = np.transpose(prediction)
    return prediction,t1,t2

'''EARS: ranking with heuristically guided ranking systems'''
def RANK(X,Y_new,Xt,Yt,elists,basis=None):
    prediction = np.zeros(np.shape(Yt))
    t1,t2 = 0,0
    for i in range(len(elists)):
        tmp_prediction,tmp_t1,tmp_t2 = RANKa(X,Y_new,Xt,Yt,elists[i],basis)
        prediction += tmp_prediction
        t1 += tmp_t1
        t2 += tmp_t2
    return prediction/len(elists),t1,t2

datasnames = ["CHD_49","Emotions","Foodtruck","GnegativeGO","HumanGO","Image","Langlog","Scene","Chess","Tmc2007_500",
    "Water_quality","Business","Entertainment","Yeast","Yelp"]
rd = ReadData()
kb = 20
misrate = 0.3
k_this = int(kb*misrate)
# mode='noise'
for mode in ('mis','part','noise'):
    for dataIdx in range(15):
        X,Y,Xt,Yt = rd.readData(dataIdx)
        Y = randmis(Y,misrate,mode)
        num_label = np.shape(Y)[1]
        if(mode=='mis'):
            Y_new = completeLabel(X,Y,k=k_this, complete_which=0, self_weight=1)
        if(mode=='part'):
            Y_new = completeLabel(X,Y,k=k_this, complete_which=1, self_weight=1)
        if(mode=='noise'):
            Y_new = completeLabel(X,Y,k=k_this, complete_which=2, self_weight=1)
        t0 = time()
        basis = basing(X,intlabel(Y_new,0),Xt)
        basis = np.array(basis.todense())
        t0 = time()-t0

        num_tree = int(round(np.sqrt(num_label)))
        edgeBegins = random.sample(range(num_label),num_tree)

        # prediction,t1,t2 = RANKa(X,Y_new,Xt,Yt,primes(getEntMat(Y_new),random.sample(range(num_label),1))[0],basis)
        prediction,t1,t2 = RANK(X,Y_new,Xt,Yt,primes(getEntMat(Y_new),edgeBegins),basis)
        # prediction,t1,t2 = RANKf(X,Y_new,Xt,Yt,basis)
        resolveResult(datasnames[dataIdx], 'EPR_'+mode, evaluate(prediction, Yt), t0+t1,t2)
