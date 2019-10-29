import numpy as np
'''
nroot = 2
#x_t1 = np.ones((5,5))
dict_x_t1 ={}

for r in range (0,2):
  for iroot in range(0,nroot):
    dict_x_t1[r,iroot] = np.zeros((5,5))
    dict_x_t1[r,iroot] += np.ones((5,5))  
    print r, iroot, dict_x_t1[r,iroot]
'''
b = np.arange(6) + 10
print b
index = np.sort(b)
print index[-1]
