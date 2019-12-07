#Import module
import pyscf.gto
from pyscf import gto

##-----Specify geometry and basis set------##

mol = pyscf.gto.M(
verbose = 5,
output = None,
unit='Bohr',
atom ='''
C   0.00000   0.000000   0.00000
H   0.00000   1.644403   1.32213
H   0.00000   -1.644403  1.32213
''', 
#basis = 'ccpvdzd',
basis = 'cc-pVDZ',
symmetry = 'c2v',
)

##------Specify linear or non-linear-------##
calc = 'ICCSD'

##------Specify convergence criteria-------##
conv = 6

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

##---------Number of roots required---------------##
nroot = [0,0,0,1]







'''
Li  0.000000,  0.000000, -0.3797714041
H   0.000000,  0.000000,  2.6437904102

O   0.000000   0.00000    0.1366052
H   0.768958   0.00000   -0.5464208
H  -0.768958   0.00000   -0.5464208
'''
