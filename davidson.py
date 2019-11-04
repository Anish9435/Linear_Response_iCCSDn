import numpy as np
import test
import inp
import math
import MP2
import trans_mo

nroot = inp.nroot

# This function finds out the orbital indices of the lowest excitation
def find_orb_indcs(Tmat):

  result = np.where(Tmat == np.amin(Tmat))
  if (len(result[0])>1):
    print 'WARNING: There is degeneracy in the system'
  ind_occ = result[0][0]
  ind_virt = result[1][0]

  return ind_occ,ind_virt


def koopmann_spectrum(occ,virt,o_act,v_act):
  
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
  return t1_tmp


def guess_X(occ,virt,o_act,v_act):


  dict_t1_nonortho_guess ={}
  dict_t2_nonortho_guess ={}
  for iroot in range(0,nroot):

    t1_guess = np.zeros((occ,virt))
    t2_guess = np.zeros((occ,occ,virt,virt))

    if(iroot==0):
      t1_tmp=koopmann_spectrum(occ,virt,o_act,v_act)

    io,iv = find_orb_indcs(t1_tmp)

    t1_guess[io,iv] = 1.0/math.sqrt(2.0)  

    dict_t1_nonortho_guess[0,iroot]=t1_guess
    dict_t2_nonortho_guess[0,iroot]=t2_guess

    t1_tmp[io,iv]=123.456

  dict_t1,dict_t2 = orthogonalize_guess(dict_t1_nonortho_guess,dict_t2_nonortho_guess)

  return dict_t1,dict_t2

def orthogonalize_guess(dict_t1_nonortho_guess,dict_t2_nonortho_guess):

  dict_t1_ortho_guess = {}
  dict_t2_ortho_guess = {}

  for iroot in range(0,nroot):
    dict_t1_ortho_guess[0,iroot] = dict_t1_nonortho_guess[0,iroot]
    dict_t2_ortho_guess[0,iroot] = dict_t2_nonortho_guess[0,iroot]

    for jroot in range(0,iroot):
      ovrlp = 2.0*np.einsum('ia,ia', dict_t1_nonortho_guess[0,iroot],dict_t1_nonortho_guess[0,jroot]) + 2.0*np.einsum('ijab,ijab',dict_t2_ortho_guess[0,iroot],dict_t2_ortho_guess[0,jroot]) - np.einsum('ijab,ijba',dict_t2_ortho_guess[0,iroot],dict_t2_ortho_guess[0,jroot])
      dict_t1_ortho_guess[0,iroot] += -ovrlp*dict_t1_nonortho_guess[0,jroot]
      dict_t2_ortho_guess[0,iroot] += -ovrlp*dict_t2_nonortho_guess[0,jroot]


  return dict_t1_ortho_guess,dict_t2_ortho_guess

def get_XO(R,D,Omega):
  X = np.divide(R,(D-Omega))
  return X
    
    
