#Import module
import pyscf.gto
from pyscf import gto

##-----Specify geometry and basis set------##
mol = pyscf.gto.M(
verbose = 4,
atom ='''
O         -1.62893       -0.04138        0.37137
H         -0.69803       -0.09168        0.09337
H         -2.06663       -0.73498       -0.13663
''',                        
basis = 'cc-pvdz',  
symmetry = True)

##------Specify linear or non-linear-------##
calc = 'ICCSD'

##------Specify convergence criteria-------##
conv = 8

##------Specify number of iteration--------##
n_iter = 200

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
n_davidson = 7
