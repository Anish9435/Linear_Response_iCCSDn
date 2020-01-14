
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                                            # Routine to Symmetrize the two body residue of ground state CC #
                                                # Author: Soumi Tribedi, Anish Chakraborty, Rahul Maitra #
                                                                  # Date - 10th Dec, 2019 # 

                   ##----------------------------------------------------------------------------------------------------------------##

##--------------------------------------------------##
          #Import important modules#
##--------------------------------------------------##

import gc
import numpy as np
import copy as cp
import MP2
import inp
import amplitude
import intermediates

##--------------------------------------------------##
          #import important parameters#
##--------------------------------------------------##

D1 = MP2.D1
D2=MP2.D2
So = MP2.So
Do=MP2.Do
Sv = MP2.Sv
Dv=MP2.Dv
conv = 10**(-inp.conv)

##--------------------------------------------------##
              #compute new t2#
##--------------------------------------------------##

def update_t2(R_ijab,t2):
  ntmax = 0
  eps_t = 100
  if eps_t >= conv:
    delt2 = np.divide(R_ijab,D2)
    t2 = t2 + delt2
  ntmax = np.size(t2)
  eps_t = float(np.sum(abs(R_ijab))/ntmax)
  return eps_t, t2, R_ijab

##--------------------------------------------------##
              #compute new t1 and t2#
##--------------------------------------------------##

def update_t1t2(R_ia,R_ijab,t1,t2):
  ntmax = 0
  eps = 100
  delt2 = np.divide(R_ijab,D2)
  delt1 = np.divide(R_ia,D1)
  t1 = t1 + delt1
  t2 = t2 + delt2
  ntmax = np.size(t1)+np.size(t2)
  eps = float(np.sum(abs(R_ia)+np.sum(abs(R_ijab)))/ntmax)
  return eps, t1, t2

##--------------------------------------------------##
                #compute new So#
##--------------------------------------------------##

def update_So(R_ijav,So):
  ntmax = 0
  eps_So = 100
  if eps_So >= conv:
    delSo = np.divide(R_ijav,Do)
    So = So + delSo
  ntmax = np.size(So)
  eps_So = float(np.sum(abs(R_ijav))/ntmax)
  return eps_So, So

##--------------------------------------------------##
                #compute new Sv#
##--------------------------------------------------##

def update_Sv(R_iuab,Sv):
  ntmax = 0
  eps_Sv = 100
  if eps_Sv >= conv:
    delSv = np.divide(R_iuab,Dv)
    Sv = Sv + delSv
  ntmax = np.size(Sv)
  eps_Sv = float(np.sum(abs(R_iuab))/ntmax)
  return eps_Sv, Sv

                          ##-----------------------------------------------------------------------------------------------------------------------##
                                                                                    #THE END#
                          ##-----------------------------------------------------------------------------------------------------------------------##
