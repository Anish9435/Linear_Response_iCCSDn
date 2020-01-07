import gc
import numpy as np
import pyscf
import inp

from pyscf import gto, scf, cc
from pyscf.cc import ccsd_t
from pyscf import symm

mol = inp.mol

# Obtain the number of atomic orbitals in the basis set
nao = mol.nao_nr()
# Obtain the number of electrons
nelec = mol.nelectron
# Compute nuclear repulsion energy
enuc = mol.energy_nuc()
# Compute one-electron kinetic integrals
T = mol.intor('cint1e_kin_sph')
# Compute one-electron potential integrals
V = mol.intor('cint1e_nuc_sph')
# Compute two-electron repulsion integrals (Chemists' notation)
v2e = mol.intor('cint2e_sph').reshape((nao,)*4)

####  Hartree-Fock  ####

mf = scf.RHF(mol).run()
E_hf = mf.e_tot
hf_mo_E = mf.mo_energy
hf_mo_coeff = mf.mo_coeff

Fock_mo = np.zeros((nao,nao))

orb_symm = []
#mol.build(0, 0, symmetry='C2v')
mo = symm.symmetrize_space(mol, mf.mo_coeff)
#orb_symm = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
orb_symm = symm.label_orb_symm_num(mol, mol.irrep_name, mol.symm_orb, mo)
#print orb_symm


#    Set up initial Fock matrix
Fock = T + V
n = nelec/2

#      Transform the 1 electron integral to MO basis
oneelecint_mo = np.einsum('ab,ac,cd->bd',hf_mo_coeff,Fock,hf_mo_coeff)

#      Transform 2 electron integral to MO Basis
twoelecint_1 = np.einsum('zs,wxyz->wxys',hf_mo_coeff,v2e)
twoelecint_2 = np.einsum('yr,wxys->wxrs',hf_mo_coeff,twoelecint_1)
twoelecint_3 = np.einsum('xq,wxrs->wqrs',hf_mo_coeff,twoelecint_2)
twoelecint_mo = np.einsum('wp,wqrs->pqrs',hf_mo_coeff,twoelecint_3)
twoelecint_1 = None
twoelecint_2 = None
twoelecint_3 = None
#      Verify integrals
E_scf_mo_1 = 0
E_scf_mo_2 = 0
E_scf_mo_3 = 0
for i in range(0,n):
  E_scf_mo_1 += oneelecint_mo[i][i]
for i in range(0,n):
  for j in range(0,n):
    E_scf_mo_2 += 2*twoelecint_mo[i][i][j][j] - twoelecint_mo[i][j][i][j]

Escf_mo = 2*E_scf_mo_1 + E_scf_mo_2 + enuc
def check_mo():
  if abs(Escf_mo - E_hf)<= 1E-6 :
    print "MO conversion successful"
  return
print Escf_mo,E_hf
print hf_mo_E
check_mo()

#Create Fock matrix
for i in range(0,nao):
  Fock_mo[i,i] = hf_mo_E[i]



gc.collect()

