# Import modules
import gc
import numpy as np
import copy as cp
import MP2

# import important stuff
occ = MP2.occ
virt = MP2.virt

#      Symmetrize R_ijab
def symmetrize(R_ijab):
  R_ijab_new = np.zeros((occ,occ,virt,virt))
  for i in range(0,occ):
        for j in range(0,occ):
                for a in range(0,virt):
                        for b in range(0,virt):
                                R_ijab_new[i,j,a,b] = R_ijab[i,j,a,b] + R_ijab[j,i,b,a]

  R_ijab = cp.deepcopy(R_ijab_new)
  #print R_ijab,"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  return R_ijab

  R_ijab = None
  R_ijab_new = None
  gc.collect()

