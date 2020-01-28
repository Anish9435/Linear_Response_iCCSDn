
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                                              # Input file to run CCSD-LRT/iCCSDn-LRT to get Excitation Energy #
                                    
                                                # Author: Anish Chakraborty, Pradipta Samanta & Rahul Maitra #
                                                                  # Date - 10th Dec, 2019 # 

                   ##----------------------------------------------------------------------------------------------------------------##


##---------------------------------------------------------##
               #Import important modules
##---------------------------------------------------------##

import pyscf.gto
from pyscf import gto

##---------------------------------------------------------##
             #Specify geometry and basis set#       
##---------------------------------------------------------##

mol = pyscf.gto.M(
verbose = 5,
output = None,
unit='Bohr',
atom ='''
O   0.000000   0.00000    0.1366052
H   0.768958   0.00000   -0.5464208
H  -0.768958   0.00000   -0.5464208
''', 
basis = 'ccpvdz',
symmetry = 'C2v',
)
##---------------------------------------------------------##
                #Specify CC-Type#
       #Specific for the ground state calculation#
       #Options are 'LCCD', 'CCSD', 'ICCSD', 
##---------------------------------------------------------##

calc = 'CCSD'

##---------------------------------------------------------##
               #Specify LRT-Type#
          #LR_type = 'ICCSD' for iCCSDn-LRT#
        #LR_type = 'CCSD' or 'None' for CCSD-LRT#
##---------------------------------------------------------##

LR_type = 'CCSD'

##---------------------------------------------------------##
            #Specify convergence criteria ground state#
##---------------------------------------------------------##

conv = 7

##---------------------------------------------------------##
     #Specify convergence criteria for excited state#
##---------------------------------------------------------##

LR_conv = 5

##---------------------------------------------------------##
      #Specify max number of iteration for ground state#
##---------------------------------------------------------##

n_iter = 30

##---------------------------------------------------------##
          #Specify max number of iteration for LRT#
##---------------------------------------------------------##

lrt_iter = 60

##---------------------------------------------------------##
                      #Specify DIIS#
     #If diis='TRUE'; max_diis needs to be specified#
       #Specific for the ground state calculation#
##---------------------------------------------------------##

diis = True
max_diis = 7

##---------------------------------------------------------##
         #Specify number of active orbitals#
    #Currently same for both ground and excited states#
##---------------------------------------------------------##

o_act = 1
v_act = 1

##---------------------------------------------------------##
         #Specify number of frozen orbitals#
##---------------------------------------------------------##

nfo = 0
nfv = 0

##---------------------------------------------------------------------------##
     #Specify no of steps after which linear combination has to be taken#
                       #Specific for LRT#
                #This might need further testing#
##---------------------------------------------------------------------------##

n_davidson = 40

##-----------------------------------------------------------------------##
             #Number of roots required for each symmetry#
      #The ordering of the states for C2v group is A1,B1,B2,A2#
      #The ordering for D2h group is Ag,B3u,B2u,B1g,B1u,B2g,B3g,Au#
                       #Specific for LRT#
##-----------------------------------------------------------------------##

nroot = [2,0,0,0]







'''
Li  0.000000,  0.000000, -0.3797714041
H   0.000000,  0.000000,  2.6437904102

O   0.00000    0.0000    -0.2214314
H   1.43043    0.0000     0.8857256
H  -1.43043    0.0000     0.8857256

! Experimental geometry
! Taken from Li and Paldus Mol. Phys. Volume 104, 2006 
O   0.000000   0.00000    0.1366052
H   0.768958   0.00000   -0.5464208
H  -0.768958   0.00000   -0.5464208

C   0.00000   0.000000   0.00000
H   0.00000   1.644403   1.32213
H   0.00000   -1.644403  1.32213

B   0.000000   0.00000    0.0000000
H   0.000000   0.00000    2.3289000

H   0.000000   0.00000    0.0000000
F   0.000000   0.00000    1.7328795

C   0.000000   0.00000   -1.1740000
C   0.000000   0.00000    1.1740000

H 0.000000 0.923274 1.238289
H 0.000000 -0.923274 1.238289
H 0.000000 0.923274 -1.238289
H 0.000000 -0.923274 -1.238289
C 0.000000 0.000000 0.668188
C 0.000000 0.000000 -0.668188

H 0.000000 0.934473 -0.588078
H 0.000000 -0.934473 -0.588078
C 0.000000 0.000000 0.000000
O 0.000000 0.000000 1.221104
'''
