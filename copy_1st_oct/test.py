import numpy as np

R_ijab = np.zeros((5,5,7,7))
t2 = np.zeros((5,5,7,7))
R_ia = np.zeros((5,7))
t1 = np.zeros((5,7))
count = 0
for i in range(0,5):
  for j in range(0,5):
    for a in range(0,7):
      for b in range(0,7):
        count= count+1
        R_ijab[i,j,a,b] = count

for i in range(0,5):
  for j in range(0,5):
    for a in range(0,7):
      for b in range(0,7):
        count= count+1
        t2[i,j,a,b] = count

for i in range(0,5):
  for a in range(0,7):
    count=count+1
    R_ia[i,a]=count

for i in range(0,5):
  for a in range(0,7):
    count=count+1
    t1[i,a]=count

t1_2 = t1.flatten()
R_ia_2 = R_ia.flatten()

print t1[0,1]
print t1_2
A = np.linalg.multi_dot([np.transpose(t1_2),R_ia_2])
#print A

#A2=np.linalg.multi_dot([np.transpose(t1),R_ia])
A2 = np.einsum('ia,ia',t1,R_ia)
#print A2

#B=np.linalg.multi_dot([np.transpose(t2),R_ijab])

