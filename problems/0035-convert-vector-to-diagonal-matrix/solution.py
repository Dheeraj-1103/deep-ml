import numpy as np

def make_diagonal(x):
    n=len(x)
    a = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(0,len(x)):
        #print(a[i][i],x[i])
        a[i][i]=x[i]
        #print(a[i][i],x[i])
    return a