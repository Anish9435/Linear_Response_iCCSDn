import numpy as np
import copy as cp

nroot = 2
B = np.zeros((nroot,nroot))

for r in range(0,5):
  B_n = np.ones((nroot*(r+1),nroot*(r+1)))
  B_n[:nroot*r,:nroot*r] = B 
  B = cp.deepcopy(B_n)
  print B
