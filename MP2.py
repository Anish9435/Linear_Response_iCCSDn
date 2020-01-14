
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                                             # Routine to calculate MP2 energy and verify with pyscf routine #
                                                # Author: Soumi Tribedi, Anish Chakraborty, Rahul Maitra #
                                                                  # Date - 10th Dec, 2019 # 

                   ##----------------------------------------------------------------------------------------------------------------##

##--------------------------------------------------##
          #Import important modules#
##--------------------------------------------------##

import numpy as np
import trans_mo
import inp
import gc
import copy as cp
import sys
from pyscf import mp

##--------------------------------------------------##
          #Import important parameters#
##--------------------------------------------------##

n = trans_mo.n
nao = trans_mo.nao
o_act = inp.o_act
v_act = inp.v_act
hf_mo_E = trans_mo.hf_mo_E
E_hf = trans_mo.E_hf
Fock_mo = trans_mo.Fock_mo
occ = n
virt = nao-n
nfo = inp.nfo
nfv = inp.nfv
twoelecint_mo = np.swapaxes(trans_mo.twoelecint_mo,1,2)  #physicist notation

##----------------------------------------------------------------------##
                  #module for the frozen orbitals#
##----------------------------------------------------------------------##

if nfo > 0:
  occ = occ - nfo
  twoelecint_mo = cp.deepcopy(twoelecint_mo[nfo:,nfo:,nfo:,nfo:])
  hf_mo_E = cp.deepcopy(hf_mo_E[nfo:])
  Fock_mo = cp.deepcopy(Fock_mo[nfo:,nfo:])
  nao = nao - nfo

if nfv > 0:
  twoelecint_mo = cp.deepcopy(twoelecint_mo[:-nfv,:-nfv,:-nfv,:-nfv])
  Fock_mo = cp.deepcopy(Fock_mo[:-nfv,:-nfv])
  hf_mo_E = hf_mo_E[:-nfv]
  nao = nao - nfv - nfo
  virt = virt - nfv  

##----------------------------------------------------------------------##
                  #Set up the denominator and t/s#
##----------------------------------------------------------------------##

D2 = np.zeros((occ,occ,virt,virt))
t2 = np.zeros((occ,occ,virt,virt))
D1 = np.zeros((occ,virt))
t1 = np.zeros((occ,virt))
Do = np.zeros((occ,occ,virt,o_act))
So = np.zeros((occ,occ,virt,o_act))
Dv = np.zeros((occ,v_act,virt,virt))
Sv = np.zeros((occ,v_act,virt,virt))

for i in range(0,occ):
  for j in range(0,occ):
    for a in range(occ,nao):
      for b in range(occ,nao):
        D2[i,j,a-occ,b-occ] = hf_mo_E[i] + hf_mo_E[j] - hf_mo_E[a] - hf_mo_E[b]
        t2[i,j,a-occ,b-occ] = twoelecint_mo[i,j,a,b]/D2[i,j,a-occ,b-occ]

if inp.calc == 'CCSD' or inp.calc == 'ICCSD' or inp.calc == 'ICCSD-PT':
  for i in range(0,occ):
    for a in range(occ,nao):
      D1[i,a-occ] = hf_mo_E[i] - hf_mo_E[a]
      t1[i,a-occ] = Fock_mo[i,a]/D1[i,a-occ]

if inp.calc == 'ICCD' or inp.calc == 'ILCCD' or inp.calc == 'ICCSD' or inp.calc == 'ICCSD-PT':
  for i in range(0,occ):
    for c in range(0,v_act):
      for a in range(0,virt):
        for b in range(0,virt):
          Dv[i,c,a,b] =  hf_mo_E[i] - hf_mo_E[c+occ] - hf_mo_E[a+occ] - hf_mo_E[b+occ]
          Sv[i,c,a,b] = twoelecint_mo[i,c+occ,a+occ,b+occ]/Dv[i,c,a,b]
  
  for i in range(0,occ):
    for j in range(0,occ):
      for a in range(0,virt):
        for k in range(occ-o_act,occ):
          Do[i,j,a,k-occ+o_act] =  hf_mo_E[i] + hf_mo_E[j] - hf_mo_E[a+occ] + hf_mo_E[k]
          So[i,j,a,k-occ+o_act] = twoelecint_mo[i,j,a+occ,k]/Do[i,j,a,k-occ+o_act]

##-------------------------------------------------------------------------------------##
                          #Calculation of MP2 energy#
##-------------------------------------------------------------------------------------##

E_mp2 = 2*np.einsum('ijab,ijab',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) - np.einsum('ijab,ijba',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
print "MP2 correlation energy is : "+str(E_mp2)
E_mp2_tot = E_hf + E_mp2
print "MP2 energy is : "+str(E_mp2_tot)

##-------------------------------------------------------------------------------------##
                          #Verify with pyscf routine#
##-------------------------------------------------------------------------------------##

m = mp.MP2(trans_mo.mf)
def check_mp2():
  if abs(m.kernel()[0]-E_mp2) <= 1E-6:
    print "MP2 successfully done"
  return

check_mp2()
gc.collect()
                          ##-----------------------------------------------------------------------------------------------------------------------##
                                                                                    #THE END#
                          ##-----------------------------------------------------------------------------------------------------------------------##
