import numpy as np

class Out():
    def __init__(self, input):
        self.input = input
    
    def predict(self, X):
        p = []
        for i in range(len(X)):
            p.append(self.input)
        return p
    
    def predict_proba(self, X):
        p = []
        for i in range(len(X)):
            p.append([1-self.input, self.input])
        return p

if __name__=="__main__":
    cls = Out(1)
    X = np.ones((201,1225))
    rst = cls.predict(X)
    print(rst)
    print(len(rst))