from mReadData import *
import random
from sklearn.neighbors import NearestNeighbors

'''produce missing/partial/noise labels: 1 for positive, -1 for negative, 0 for missing'''
def randmis(mat_, ratio, mode='mis'): # mode={'miss','part','noise'}
    x,y = np.shape(mat_)
    sum = x*y
    mat_ = (mat_-0.5)*2
    mat2 = np.array(mat_)
    idx = random.sample(range(0,sum),int(sum*ratio))
    for i in range(len(idx)):
        a = int(idx[i]/y)
        b = idx[i] % y
        if(mode=='mis'):
            mat2[a][b] = 0
        elif(mode=='part'):
            mat2[a][b] = 1
        elif(mode=='noise'):
            mat2[a][b] *= -1
        else:
            print('ERROR')
    mat2 = keep1(mat_,mat2)
    return mat2

'''for each label, if all positive labels are deleted, revise one of them'''
def keep1(mat_, mat2):
    for j in range(len(mat_[0])):
        if(np.sum(mat2[:,j]==1) == 0):
            idx = np.argwhere(mat_[:,j]==1).flatten()
            if(len(idx)==0):
                continue
            select_a = random.randint(0,len(idx)-1)
            mat2[idx[select_a]][j] = 1
        if(np.sum(mat2[:,j]==-1) == 0):
            idx = np.argwhere(mat_[:,j]==-1).flatten()
            if(len(idx)==0):
                continue
            select_a = random.randint(0,len(idx)-1)
            mat2[idx[select_a]][j] = -1
    return mat2

'''semi-completion: simply sum(neighbor)/k'''
def completeLabel2(X, Y, k=10, complete_which=0):
    findNb = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    findNb.fit(X)
    indices = findNb.kneighbors(X, return_distance=False)
    label = np.array(Y).astype(np.float16)
    for i in range(len(Y)):
        indx = indices[i]
        for j in range(len(Y[0])):
            if(Y[i][j]==complete_which):
                label[i][j]=np.sum(Y[indx,j])/k
    return label
'''semi-completion: sum(neighbor)/[k-num(mis)]'''
def completeLabel(X, Y, k=10, complete_which=0, self_weight=0):
    findNb = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    findNb.fit(X)
    indices = findNb.kneighbors(X, return_distance=False)
    label = np.array(Y).astype(np.float16)
    for i in range(len(Y)):
        indx = indices[i]
        for j in range(len(Y[0])):
            if(complete_which==2):
                label[i][j]=np.sum(Y[indx,j]+label[i][j]*self_weight)/(k+self_weight)
            if(Y[i][j]==complete_which):
                k_this = np.sum(np.abs(Y[indx,j]))
                if(k_this==0):
                    label[i][j]=complete_which
                else:
                    label[i][j]=np.sum(Y[indx,j]+complete_which*self_weight)/(k_this+self_weight)
    return label

'''convert numeric label to int label: 1 for positive, 0 for negative'''
def intlabel(labelorg, threshold=0.5):
    N,Q = np.shape(labelorg)
    label = np.zeros((N,Q))
    for i in range(N):
        for j in range(Q):
            if(labelorg[i][j]>=threshold):
                label[i][j]=1
            else:
                label[i][j]=0
    return label

def fill1(Y):
    Y = np.array(Y)
    for j in range(np.shape(Y)[1]):
        if(np.sum(Y[:,j])==0):
            Y[0][j] = 1
    return Y

if __name__=="__main__":
    datasnames = ["Birds","CAL500","CHD_49","Enron","Flags","Foodtruck",
        "Genbase","GnegativeGO","GpositiveGO","Image","Langlog","Medical","PlantGO","Scene","Slashdot","Chemistry","Chess","Coffee","VirusGO","Yeast","Yelp"]
    rd = ReadData(datas=datasnames)
    
    for dataIdx in range(5,6):
        # print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        Y = randmis(Y,0.3)
