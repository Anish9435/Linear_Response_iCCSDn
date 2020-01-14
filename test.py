import numpy as np
B=np.matrix([[11, 2, 3 ,2],[3, 14, 4, 5], [7, 8, 15, 9], [6, 5, 8, 17]]) 
         
w,vect = np.linalg.eig(B)
print w
print vect.real
for n in range (0,len(w)):
  m = w.argsort()[n]
  print m
  print vect[:,m].real


#w_sort = np.sort(w)
#print w_sort
#
#for n in range (0,len(w_sort)+1):
#  m = w_sort.argsort()[:n]
#print m
