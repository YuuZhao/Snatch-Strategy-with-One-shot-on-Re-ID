import  numpy as np

n = np.zeros((5))
n[3]=1
n[4]=3
print(len(np.nonzero(n)[0]))