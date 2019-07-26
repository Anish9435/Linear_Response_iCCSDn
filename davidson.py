import numpy as np
import test
import inp

def guess_X(occ,virt,o_act,v_act):
  t1_guess = np.zeros((occ,virt))
  t1_guess[2,1] = 1.0
  t2_guess = np.zeros((occ,occ,virt,virt))
  So_guess = np.zeros((occ,occ,virt,o_act))
  Sv_guess = np.zeros((occ,v_act,virt,virt))
  return t1_guess,t2_guess,So_guess,Sv_guess

def t1_contribution(t1,R_ia):
  w1=np.einsum('ia,ia',t1,R_ia)
  return w1
  
def t2_contribution(t2,R_ijab):
  w2=np.einsum('ijab,ijab',t2,R_ijab)
  return w2

def So_contribution(So,R_ijav):
  w3=np.einsum('ijav,ijav',So,R_ijav)
  return w3

def Sv_contribution(Sv,R_iuab):
  w4=np.einsum('iuab,iuab',Sv,R_iuab)
  return w4

def get_w(w1,w2,w3,w4):
  w = w1+w2+w3+w4
  return w

def get_residue(AX,w,X):  #formation of residual matrix
  R = AX - w*X 
  return R

def get_X(R,D):
  X = np.divide(R,D)
  return X
 
  
