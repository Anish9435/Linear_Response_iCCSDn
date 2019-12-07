import numpy as np
import test
import inp
import math
import MP2
import trans_mo


_multd2h = [[1,2,3,4,5,6,7,8],
[2,1,4,3,6,5,8,7],
[3,4,1,2,7,8,5,6],
[4,3,2,1,8,7,6,5],
[5,6,7,8,1,2,3,4],
[6,5,8,7,2,1,4,3],
[7,8,5,6,3,4,1,2],
[8,7,6,5,4,3,2,1]]

# This function finds out the orbital indices of the lowest excitation
def find_orb_indcs(Tmat):

  result = np.where(Tmat == np.amin(Tmat))
  if (len(result[0])>1):
    print 'WARNING: There is degeneracy in the system'
  ind_occ = result[0][0]
  ind_virt = result[1][0]
  
  return ind_occ,ind_virt

def find_orb_indcs_all(T1mat, T2mat):

  iEx = 0

  min_val_1 = np.amin(T1mat)
  result = np.where(T1mat == np.amin(T1mat))

  min_val_2 = np.amin(T2mat)
  print 'DEBUGGING', min_val_1, min_val_2
  if (min_val_1 <= min_val_2):
    if (len(result[0])>1):
      print 'WARNING: There is degeneracy in the system'
    ind_occ = result[0][0]
    ind_virt = result[1][0]
    iEx = 1
  else:
    print 'Double excitation dominated guess'
    result = np.where(T2mat == np.amin(T2mat))
    if (len(result[0])>1):
      print 'WARNING: There is degeneracy in the system'
    ind_occ = [result[0][0], result[1][0]]
    ind_virt = [result[2][0], result[3][0]]
    iEx = 2

  
  return ind_occ,ind_virt, iEx



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

def koopmann_spectrum_sym_sing(occ,virt,o_act,v_act, orb_sym, isym):
  
  Fock_mo = MP2.Fock_mo
  # Store the excitation energy obtained using the Fock matrix
  t1_tmp = np.zeros((occ,virt))
  for i in range(0,occ):
    sym_i = orb_sym[i]
    for a in range(0,virt):
      sym_a = orb_sym[a+occ]
      prod_sym = _multd2h[sym_i][sym_a] - 1
      # Store only the symmetry allowed excitations
      if (prod_sym == isym):
        t1_tmp[i,a] = abs(Fock_mo[i,i]-Fock_mo[a+occ,a+occ])
      else: 
        t1_tmp[i,a] = 123.456 # initialize with some large number

  return t1_tmp

def koopmann_spectrum_sym_doub(occ,virt,o_act,v_act, orb_sym, isym):
  
  Fock_mo = MP2.Fock_mo
  # Store the excitation energy obtained using the Fock matrix
  t2_tmp = np.zeros((occ,occ,virt,virt))
  for i in range(0,occ):
    sym_i = orb_sym[i]
    for j in range(0,occ):
      sym_j = orb_sym[j]
      sym_ij = _multd2h[sym_i][sym_j] - 1
      for a in range(0,virt):
        sym_a = orb_sym[a+occ]
        sym_ija = _multd2h[sym_ij][sym_a] - 1
        for b in range(0,virt):
          sym_b = orb_sym[b+occ]
          prod_sym = _multd2h[sym_ija][sym_b] - 1
#      print 'Debug:', i, a+occ
#      print 'Debug:', sym_i, sym_a, prod_sym, isym
      # Store only the symmetry allowed excitations
          if (prod_sym == isym):
            t2_tmp[i,j,a,b] = abs(Fock_mo[i,i] + Fock_mo[j,j]-Fock_mo[a+occ,a+occ]-Fock_mo[b+occ,b+occ])
          else: 
            t2_tmp[i,j,a,b] = 123.456 # initialize with some large number

 # print '****'
 # print t1_tmp
 # print '****'
  return t2_tmp


def guess_X(occ,virt,o_act,v_act,nroot):

  dict_t1 = {}
  dict_t2 = {}
  dict_So = {}
  dict_Sv = {}

  for iroot in range(0,nroot):
    t1_guess = np.zeros((occ,virt))
    t2_guess = np.zeros((occ,occ,virt,virt))
    So_guess = np.zeros((occ,occ,virt,o_act))
    Sv_guess = np.zeros((occ,v_act,virt,virt))

    if(iroot==0):
      t1_tmp=koopmann_spectrum(occ,virt,o_act,v_act)

    io,iv = find_orb_indcs(t1_tmp)

    t1_guess[io,iv] = 1.0/math.sqrt(2.0)  

    dict_t1[0,iroot] = t1_guess
    dict_t2[0,iroot] = t2_guess
    dict_So[0,iroot] = So_guess
    dict_Sv[0,iroot] = Sv_guess

    t1_tmp[io,iv]=123.456

  return dict_t1,dict_t2,dict_So,dict_Sv

def guess_sym(occ,virt,o_act,v_act, orb_sym, isym, nroot):

  dict_t1 = {}
  dict_t2 = {}
  dict_So = {}
  dict_Sv = {}

  for iroot in range(0,nroot):
    t1_guess = np.zeros((occ,virt))
    t2_guess = np.zeros((occ,occ,virt,virt))
    So_guess = np.zeros((occ,occ,virt,o_act))
    Sv_guess = np.zeros((occ,v_act,virt,virt))

    if(iroot==0):
      t1_tmp=koopmann_spectrum_sym_sing(occ,virt,o_act,v_act, orb_sym, isym)
      t2_tmp=koopmann_spectrum_sym_doub(occ,virt,o_act,v_act, orb_sym, isym)

    io,iv,iEx = find_orb_indcs_all(t1_tmp, t2_tmp)

    if (iEx == 1):
      t1_guess[io,iv] = 1.0/math.sqrt(2.0)  
      t1_tmp[io,iv]=123.456
    elif (iEX == 2):
      t2_guess[io[0],io[1],iv[0],iv[1]] = 1.0/2.0  
      t2_tmp[io[0],io[1],iv[0],iv[1]]=123.456
    else: 
      print 'Wrong Guess'
      exit()     

    dict_t1[0,iroot] = t1_guess
    dict_t2[0,iroot] = t2_guess
    dict_So[0,iroot] = So_guess
    dict_Sv[0,iroot] = Sv_guess


  return dict_t1,dict_t2,dict_So,dict_Sv

def guess_X_man(occ,virt,o_act,v_act,nroot):

  dict_t1_nonortho_guess = {}
  dict_t2_nonortho_guess = {}
  dict_So_nonortho_guess = {}
  dict_Sv_nonortho_guess = {}

  for iroot in range(0,nroot):
    t1_guess = np.zeros((occ,virt))
    t2_guess = np.zeros((occ,occ,virt,virt))
    So_guess = np.zeros((occ,occ,virt,o_act))
    Sv_guess = np.zeros((occ,v_act,virt,virt))

    if(iroot==0):
      t1_tmp=koopmann_spectrum(occ,virt,o_act,v_act)

    io,iv = find_orb_indcs(t1_tmp)

    t1_guess[io,iv] = 1.0/math.sqrt(2.0)  

    dict_t1_nonortho_guess[0,iroot] = t1_guess
    dict_t2_nonortho_guess[0,iroot] = t2_guess
    dict_So_nonortho_guess[0,iroot] = So_guess
    dict_Sv_nonortho_guess[0,iroot] = Sv_guess

    t1_tmp[io,iv]=123.456

  return dict_t1,dict_t2,dict_So,dict_Sv

def get_XO(R,D,Omega):
  X = np.divide(R,(D-Omega))
  return X
    
def get_orb_sym(orb_sym_pyscf, sym):

  sym_num_comm = {}
  if (sym=='D2h'):
    sym_num_comm[0] = 0
    sym_num_comm[1] = 3
    sym_num_comm[2] = 5
    sym_num_comm[3] = 6
    sym_num_comm[4] = 7
    sym_num_comm[5] = 4
    sym_num_comm[6] = 2
    sym_num_comm[7] = 1
  if ((sym=='C2v') or (sym=='C2h')):
    sym_num_comm[0] = 0
    sym_num_comm[1] = 3
    sym_num_comm[2] = 1
    sym_num_comm[3] = 2

  orb_sym = []
  for i in orb_sym_pyscf:
    orb_sym.append(sym_num_comm[i])
    
  return orb_sym
