#Import module
import pyscf.gto
from pyscf import gto

##-----Specify geometry and basis set------##

mol = pyscf.gto.M(
verbose = 5,
output = None,
unit='Bohr',
atom = [['Li',(  0.000000,  0.000000, -0.3797714041)],
        ['H',(  0.000000,  0.000000,  2.6437904102)]],
basis = {'H': 'sto-3g','Li':'sto-3g'},
symmetry = True)

##------Specify linear or non-linear-------##
calc = 'CCSD'

##------Specify convergence criteria-------##
conv = 8

##------Specify number of iteration--------##
n_iter = 40

##------Specify number of iteration for LRT--------##
lrt_iter = 30

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

##---------Number of roots required---------------##
nroot = 2
