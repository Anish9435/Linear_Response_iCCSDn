import numpy as np
import test
import inp
import math
import MP2
import trans_mo

# This function finds out the orbital indices of the lowest excitation
def find_orb_indcs(Tmat):

  result = np.where(Tmat == np.amin(Tmat))
  if (len(result[0])>1):
    print 'WARNING: There is degeneracy in the system'
  ind_occ = result[0][0]
  ind_virt = result[1][0]

  return ind_occ,ind_virt

def guess_X(occ,virt,o_act,v_act):

  t1_guess = np.zeros((occ,virt))
  t2_guess = np.zeros((occ,occ,virt,virt))
  So_guess = np.zeros((occ,occ,virt,o_act))
  Sv_guess = np.zeros((occ,v_act,virt,virt))

  # Need both the fock matrix and the one-body hamiltonian matrix to find out lowest excitation
  Fock_mo = MP2.Fock_mo
  h_pq = trans_mo.oneelecint_mo
  # Store the excitation energy obtained using the Fock matrix
  t1_tmp = np.zeros((occ,virt))
  for i in range(0,occ):
    for a in range(0,virt):
      # Store only the symmetry allowed excitations
      if (abs(h_pq[i,a+occ])>1e-12):
        t1_tmp[i,a] = abs(Fock_mo[i,i]-Fock_mo[a+occ,a+occ])
      else: 
        t1_tmp[i,a] = 123.456 # initialize with some large number

  io,iv = find_orb_indcs(t1_tmp)

  # Need to normalize the guess vector
  t1_guess[io,iv] = 1.0/math.sqrt(2.0)

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
 
def get_XO(R,D,Omega):
  X = np.divide(R,(D-Omega))
  return X
  
