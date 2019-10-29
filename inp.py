#Import module
import pyscf.gto
from pyscf import gto

##-----Specify geometry and basis set------##

mol = pyscf.gto.M(
verbose = 5,
output = None,
unit='Bohr',
atom ='''
O       0.0000000       0.0000000       0.1366050
H       0.0000000       0.7689580       -0.5464210
H       0.0000000       -0.7689580      -0.5464210
''', 
basis = 'cc-pVDZ',       
symmetry = True)

##------Specify linear or non-linear-------##
calc = 'CCSD'

##------Specify convergence criteria-------##
conv = 8

##------Specify number of iteration--------##
n_iter = 40

##------Specify DIIS-------------##
diis = True
max_diis = 7

##------Specify number of active orbitals----##
o_act = 2
v_act = 2

##-----specify number of frozen orbitals-------##
nfo = 0
nfv = 0

##-----Specify no of steps after which linear combination has to be taken-----##
n_davidson = 40

##-----Projecting out the ground state t and s i.e coupled cluster solution---------## 
proj_out_t0 = True

##------Number of roots required-------------##
nroot = 2
