                   
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                                        # Routine to calculate the Excitation Energy for CCSD/iCCSDn methodology #
                                    
                              # This code runs for the ground state first and from that one can get converged coupled #
                              # cluster t and s. Then using the first order t and s we can calculate property and the #
                              # properties are dependent on the external perturbation given to the system. The code   #
                              # generates different symmetry excitations as requested in the input                    #

                                         # Author: Anish Chakraborty, Pradipta Samanta & Rahul Maitra #
                                                           # Date - 10th Dec, 2019 # 

                   ##----------------------------------------------------------------------------------------------------------------##


##---------------------------------------------------------------------------##
                       #Import important modules#
##---------------------------------------------------------------------------##

import numpy as np
import copy as cp
import davidson
import inp
import MP2
import trans_mo
import intermediates
import intermediates_response
import amplitude
import amplitude_response
import cc_symmetrize
import cc_update
import main
import math
import scipy

##---------------------------------------------------------------------------##
                     #Import important functionalities/variables#
                         #and conditioning for iCCSD method#
##---------------------------------------------------------------------------##

tiCCSD = False
if inp.LR_type == 'ICCSD':
  tiCCSD = True

t1 = main.t1
t2 = main.t2
occ = MP2.occ
virt = MP2.virt
nao = MP2.nao
o_act = inp.o_act
v_act = inp.v_act
n_iter = inp.n_iter
lrt_iter = inp.lrt_iter
n_davidson = inp.n_davidson
eta = inp.eta
hf_mo_E = trans_mo.hf_mo_E
D1 = 1.0*MP2.D1 + eta
D2 = 1.0*MP2.D2 + eta


if (tiCCSD):
  So = main.So
  Sv = main.Sv
  Do = 1.0*MP2.Do + eta
  Dv = 1.0*MP2.Dv + eta

conv = 4*(10**(-inp.LR_conv))

sym = trans_mo.mol.symmetry
orb_sym_pyscf = MP2.orb_symm
orb_sym = davidson.get_orb_sym(orb_sym_pyscf, sym)

##---------------------------------------------------------------------------------##
                    #Function to calculate Excitation Energy# 
##---------------------------------------------------------------------------------##

def calc_excitation_energy(isym, nroot):
  
  if (tiCCSD):
    dict_t1,dict_t2,dict_So,dict_Sv = davidson.guess_sym(occ,virt,o_act,v_act,orb_sym, isym, nroot)
  else:
    dict_t1,dict_t2 = davidson.guess_sym(occ,virt,o_act,v_act,orb_sym, isym, nroot)

  twoelecint_mo = MP2.twoelecint_mo 
  count = [0]*nroot
  
##----------------------------------------------------------------------------------##
              #Initialization of Dictionary for storing values#
##----------------------------------------------------------------------------------##
  
  dict_Y_ia = {}
  dict_Y_ijab = {}

  if (tiCCSD):
    dict_Y_ijav = {}
    dict_Y_iuab = {}
  
##--------------------------------------------------------------------------------------##
           #Initialization of the nroot*nroot B matrix for zeroth iteration#
##--------------------------------------------------------------------------------------##
  
  B_Y_ia = np.zeros((nroot,nroot))
  B_Y_ijab = np.zeros((nroot,nroot))

  if (tiCCSD):
    B_Y_ijav = np.zeros((nroot,nroot))
    B_Y_iuab = np.zeros((nroot,nroot))
  
##-----------------------------------------------------------------##
                      #Iteration begins#

##-----------------------------------------------------------------##
  print ("---------------------------------------------------------")
  print ("               Molecular point group   "+str(sym))
  print ("    Linear Response iteration begins for symmetry   "+str(isym))
  print ("---------------------------------------------------------")
  for x in range(0,lrt_iter):
  
##-----------------------------------------------------------------##
              #conditioning with the remainder#
##-----------------------------------------------------------------##
  
    print ("")
    print ("-------------------------------------------------")
    print ("          Iteration number "+str(x))
    r = x%n_davidson
    print ("          Subspace vector "+str(r))
    print ("-------------------------------------------------")
  
    if(x>0):
      if r==0:
        dict_Y_ia.clear()
        dict_Y_ijab.clear()

        if (tiCCSD):
          dict_Y_ijav.clear()
          dict_Y_iuab.clear()
  
        dict_t1.clear()
        dict_t2.clear()

        if (tiCCSD):
          dict_So.clear()
          dict_Sv.clear()

        B_Y_ia = np.zeros((nroot*(r+1),nroot*(r+1)))
        B_Y_ijab = np.zeros((nroot*(r+1),nroot*(r+1)))

        if (tiCCSD):
          B_Y_ijav = np.zeros((nroot*(r+1),nroot*(r+1)))
          B_Y_iuab = np.zeros((nroot*(r+1),nroot*(r+1)))

        for iroot in range(0,nroot):
          dict_t1[r,iroot] =  dict_x_t1[n_davidson-1,iroot] 
          dict_t2[r,iroot] =  dict_x_t2[n_davidson-1,iroot]
  
          if (tiCCSD):
            dict_So[r,iroot] =  dict_x_So[n_davidson-1,iroot]
            dict_Sv[r,iroot] =  dict_x_Sv[n_davidson-1,iroot]

    for iroot in range(0,nroot):
  
##-----------------------------------------------------------------------------------------------------##
                       #Intermediate Diagram formation for Linear Response Theory#
##-----------------------------------------------------------------------------------------------------##
   
      # Diagrams and intermediates of coupled cluster theory i.e AX with new t and s-
      I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = intermediates_response.initialize() 
  
##------------------------------------------------------------------------------------------------------##
                            # Linear terms of both R_ia and R_ijab
##------------------------------------------------------------------------------------------------------##
  
      dict_Y_ia[r,iroot] = amplitude_response.singles_response_linear(I_oo,I_vv,dict_t1[r,iroot],dict_t2[r,iroot])  
      dict_Y_ijab[r,iroot] = amplitude_response.doubles_response_linear(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,dict_t1[r,iroot],dict_t2[r,iroot])
  
##------------------------------------------------------------------------------------------------------##
                        #update and generation of the new intermediates
##------------------------------------------------------------------------------------------------------##

      I1, I2 = intermediates_response.R_ia_intermediates(t1)
      I1_new, I2_new = intermediates_response.R_ia_intermediates(dict_t1[r,iroot])  
      I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = intermediates_response.update_int_response(t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)
      I_oo_new,I_vv_new,Ioooo_new,Iovvo_new,Iovvo_2_new,Iovov_new = intermediates_response.update_int_response(dict_t2[r,iroot],I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)
  
##-------------------------------------------------------------------------------------------------------##
                                      #one body diagrams#
##-------------------------------------------------------------------------------------------------------##
      
      dict_Y_ia[r,iroot] += amplitude_response.singles_response_quadratic(I_oo,I_vv,I1,I2,dict_t1[r,iroot],dict_t2[r,iroot])
      dict_Y_ia[r,iroot] += amplitude_response.singles_response_quadratic(I_oo_new,I_vv_new,I1_new,I2_new,t1,t2)
  
      dict_Y_ia[r,iroot] += -2*np.einsum('ibkj,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])       #diagram non-linear a
      dict_Y_ia[r,iroot] +=  2*np.einsum('cbaj,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,dict_t1[r,iroot]) #diagram non-linear c
      dict_Y_ia[r,iroot] +=  np.einsum('ibjk,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])         #diagram non-linear b
      dict_Y_ia[r,iroot] += -np.einsum('cbja,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1,dict_t1[r,iroot])   #diagram non-linear d
    
      dict_Y_ia[r,iroot] += -2*np.einsum('ibkj,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ia[r,iroot] +=  2*np.einsum('cbaj,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],dict_t1[r,iroot],t1)
      dict_Y_ia[r,iroot] +=  np.einsum('ibjk,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ia[r,iroot] += -np.einsum('cbja,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],dict_t1[r,iroot],t1)
  
##---------------------------------------------------------------------------------------------------------##
                                      #higher order one-body terms#
##------------------------------------------------------------------------------------------------------##
      '''
                         ##--------------------------------------##
                                ##diagram non-linear m## 
                         ##--------------------------------------##

      int_one_body = np.einsum('cdkl,kd,la->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
      dict_Y_ia[r,iroot] += np.einsum('ca,ic->ia',int_one_body,dict_t1[r,iroot])
      int_one_body = np.einsum('cdkl,kd,la->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
      dict_Y_ia[r,iroot] += np.einsum('ca,ic->ia',int_one_body,t1)
      int_one_body = np.einsum('cdkl,kd,la->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ia[r,iroot] += np.einsum('ca,ic->ia',int_one_body,t1)

                         ##--------------------------------------##
                                ##diagram non-linear n## 
                         ##--------------------------------------##

      int_one_body2 = 2.0*np.einsum('dclk,ld,ic->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
      dict_Y_ia[r,iroot] += -np.einsum('ik,ka->ia',int_one_body2,dict_t1[r,iroot])
      int_one_body2 = 2.0*np.einsum('dclk,ld,ic->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
      dict_Y_ia[r,iroot] += -np.einsum('ik,ka->ia',int_one_body2,t1)
      int_one_body2 = 2.0*np.einsum('dclk,ld,ic->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ia[r,iroot] += -np.einsum('ik,ka->ia',int_one_body2,t1)
      '''
##---------------------------------------------------------------------------------------------------------##
                               #Further update of the intermdiates#
##---------------------------------------------------------------------------------------------------------##
  
      I_oo,I_vv,I_oovo,I_vovv = intermediates_response.singles_intermediates_response(t1,t2,I_oo,I_vv)
      I_oo_new,I_vv_new,I_oovo_new,I_vovv_new = intermediates_response.singles_intermediates_response(dict_t1[r,iroot],dict_t2[r,iroot],I_oo_new,I_vv_new)
  
##---------------------------------------------------------------------------------------------------------##
                                      #two body diagrams#
##---------------------------------------------------------------------------------------------------------##
  
      dict_Y_ijab[r,iroot] += amplitude_response.doubles_response_quadratic(I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov,dict_t2[r,iroot]) 
      dict_Y_ijab[r,iroot] += amplitude_response.doubles_response_quadratic(I_oo_new,I_vv_new,Ioooo_new,Iovvo_new,Iovvo_2_new,Iovov_new,t2) 
      dict_Y_ijab[r,iroot] += amplitude_response.singles_n_doubles_response(dict_t1[r,iroot],I_oovo,I_vovv)
      dict_Y_ijab[r,iroot] += amplitude_response.singles_n_doubles_response(t1,I_oovo_new,I_vovv_new)
  
      dict_Y_ijab[r,iroot] += 0.5*np.einsum('ijkl,ka,lb->ijab',twoelecint_mo[:occ,:occ,:occ,:occ],t1,dict_t1[r,iroot])      #diagram non-linear 1
      dict_Y_ijab[r,iroot] += 0.5*np.einsum('cdab,ic,jd->ijab',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],t1,dict_t1[r,iroot])  #diagram non-linear 2
      dict_Y_ijab[r,iroot] += -np.einsum('ickb,ka,jc->ijab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,dict_t1[r,iroot])   #diagrams non-linear 3
      dict_Y_ijab[r,iroot] += -np.einsum('icak,jc,kb->ijab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,dict_t1[r,iroot])   #diagrams non-linear 4
    
      dict_Y_ijab[r,iroot] += 0.5*np.einsum('ijkl,ka,lb->ijab',twoelecint_mo[:occ,:occ,:occ,:occ],dict_t1[r,iroot],t1)      #diagram non-linear 1
      dict_Y_ijab[r,iroot] += 0.5*np.einsum('cdab,ic,jd->ijab',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],dict_t1[r,iroot],t1)  #diagram non-linear 2
      dict_Y_ijab[r,iroot] += -np.einsum('ickb,ka,jc->ijab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],dict_t1[r,iroot],t1)   #diagrams non-linear 3
      dict_Y_ijab[r,iroot] += -np.einsum('icak,jc,kb->ijab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],dict_t1[r,iroot],t1)   #diagrams non-linear 4
  
##---------------------------------------------------------------------------------------------------------##
                                      #higher order terms#
##---------------------------------------------------------------------------------------------------------##
      '''   
                         ##--------------------------------------##
                                ##diagram non-linear 30## 
                         ##--------------------------------------##

      int1 = -np.einsum('iclk,la,jc->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,t1)  
      dict_Y_ijab[r,iroot] += -np.einsum('ijak,kb->ijab',int1,dict_t1[r,iroot])  
      int1 = -np.einsum('iclk,la,jc->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])  
      dict_Y_ijab[r,iroot] += -np.einsum('ijak,kb->ijab',int1,t1)  
      int1 = -np.einsum('iclk,la,jc->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)  
      dict_Y_ijab[r,iroot] += -np.einsum('ijak,kb->ijab',int1,t1)  
      int1 = None
                
                         ##--------------------------------------##
                                 ##diagram non-linear 31## 
                         ##--------------------------------------##

      int2 = np.einsum('cdbk,jc,id->jibk',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,t1)
      dict_Y_ijab[r,iroot] += -np.einsum('jibk,ka->ijab',int2,dict_t1[r,iroot])
      int2 = np.einsum('cdbk,jc,id->jibk',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,dict_t1[r,iroot])
      dict_Y_ijab[r,iroot] += -np.einsum('jibk,ka->ijab',int2,t1)
      int2 = np.einsum('cdbk,jc,id->jibk',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],dict_t1[r,iroot],t1)
      dict_Y_ijab[r,iroot] += -np.einsum('jibk,ka->ijab',int2,t1)
      int2 = None

                         ##--------------------------------------##
                                 ##diagram non-linear 32&33## 
                         ##--------------------------------------##

      int3 = -np.einsum('dclk,ic,ka->dila',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
      dict_Y_ijab[r,iroot] += 2*np.einsum('dila,jlbd->ijab',int3,dict_t2[r,iroot]) #32
      dict_Y_ijab[r,iroot] += -np.einsum('dila,jldb->ijab',int3,dict_t2[r,iroot])  #33
      int3 = -np.einsum('dclk,ic,ka->dila',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
      dict_Y_ijab[r,iroot] += 2*np.einsum('dila,jlbd->ijab',int3,t2)
      dict_Y_ijab[r,iroot] += -np.einsum('dila,jldb->ijab',int3,t2)
      int3 = -np.einsum('dclk,ic,ka->dila',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ijab[r,iroot] += 2*np.einsum('dila,jlbd->ijab',int3,t2)
      dict_Y_ijab[r,iroot] += -np.einsum('dila,jldb->ijab',int3,t2)
      int3 = None

                         ##--------------------------------------##
                                 ##diagram non-linear 34## 
                         ##--------------------------------------##
     
      int4 = np.einsum('dckl,kc,lb->db',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1) 
      dict_Y_ijab[r,iroot] += np.einsum('db,jida->ijab',int4,dict_t2[r,iroot])
      int4 = np.einsum('dckl,kc,lb->db',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot]) 
      dict_Y_ijab[r,iroot] += np.einsum('db,jida->ijab',int4,t2)
      int4 = np.einsum('dckl,kc,lb->db',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1) 
      dict_Y_ijab[r,iroot] += np.einsum('db,jida->ijab',int4,t2)
      int4 = None

                         ##--------------------------------------##
                                 ##diagram non-linear 35## 
                         ##--------------------------------------##

      int5 = -np.einsum('cdlk,kc,jd->jl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
      dict_Y_ijab[r,iroot] += -np.einsum('jl,liba->ijab',int5,dict_t2[r,iroot])
      int5 = -np.einsum('cdlk,kc,jd->jl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
      dict_Y_ijab[r,iroot] += -np.einsum('jl,liba->ijab',int5,t2)
      int5 = -np.einsum('cdlk,kc,jd->jl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ijab[r,iroot] += -np.einsum('jl,liba->ijab',int5,t2)
      int5 = None

                         ##--------------------------------------##
                                 ##diagram non-linear 36## 
                         ##--------------------------------------##

      int6 = -np.einsum('cdlk,jc,ka->jdla',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
      dict_Y_ijab[r,iroot] += -np.einsum('jdla,libd->ijab',int6,dict_t2[r,iroot])
      int6 = -np.einsum('cdlk,jc,ka->jdla',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
      dict_Y_ijab[r,iroot] += -np.einsum('jdla,libd->ijab',int6,t2)
      int6 = -np.einsum('cdlk,jc,ka->jdla',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ijab[r,iroot] += -np.einsum('jdla,libd->ijab',int6,t2)
      int6 = None

                         ##--------------------------------------##
                                 ##diagram non-linear 37## 
                         ##--------------------------------------##

      int7 = 0.5*np.einsum('cdlk,jc,id->jilk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
      dict_Y_ijab[r,iroot] += np.einsum('jilk,lkba->ijab',int7,dict_t2[r,iroot])
      int7 = 0.5*np.einsum('cdlk,jc,id->jilk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
      dict_Y_ijab[r,iroot] += np.einsum('jilk,lkba->ijab',int7,t2)
      int7 = 0.5*np.einsum('cdlk,jc,id->jilk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ijab[r,iroot] += np.einsum('jilk,lkba->ijab',int7,t2)
      int7 = None

                         ##--------------------------------------##
                                 ##diagram non-linear 38## 
                         ##--------------------------------------##

      int8 = np.einsum('cdlk,lb,ka->cdba',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
      dict_Y_ijab[r,iroot] += 0.5*np.einsum('cdba,jicd->ijab',int8,dict_t2[r,iroot]) 
      int8 = np.einsum('cdlk,lb,ka->cdba',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
      dict_Y_ijab[r,iroot] += 0.5*np.einsum('cdba,jicd->ijab',int8,t2) 
      int8 = np.einsum('cdlk,lb,ka->cdba',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ijab[r,iroot] += 0.5*np.einsum('cdba,jicd->ijab',int8,t2) 
      int8 = None

                         ##--------------------------------------##
                                 ##diagram non-linear 39## 
                         ##--------------------------------------##

      int9 = -np.einsum('cdkl,jc,lb->jdkb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
      dict_Y_ijab[r,iroot] += -np.einsum('jdkb,kida->ijab',int9,dict_t2[r,iroot])
      int9 = -np.einsum('cdkl,jc,lb->jdkb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
      dict_Y_ijab[r,iroot] += -np.einsum('jdkb,kida->ijab',int9,t2)
      int9 = -np.einsum('cdkl,jc,lb->jdkb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ijab[r,iroot] += -np.einsum('jdkb,kida->ijab',int9,t2)
      int9 = None

                         ##--------------------------------------##
                                 ##diagram non-linear 40## 
                         ##--------------------------------------##

      int10 = -0.5*np.einsum('cdkl,kb,id->cibl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot]) 
      int10 += -0.5*np.einsum('cdkl,kb,id->cibl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1) 
      dict_Y_ijab[r,iroot] += -np.einsum('cibl,jc,la->ijab',int10,t1,t1)
      int10 = -0.5*np.einsum('cdkl,kb,id->cibl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1) 
      dict_Y_ijab[r,iroot] += -np.einsum('cibl,jc,la->ijab',int10,dict_t1[r,iroot],t1)
      dict_Y_ijab[r,iroot] += -np.einsum('cibl,jc,la->ijab',int10,t1,dict_t1[r,iroot])
      int10 = None

                         ##--------------------------------------##
                                 ##diagram non-linear 38'## 
                         ##--------------------------------------##

      int11 = 2.0*np.einsum('dclk,ld,ic->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
      dict_Y_ijab[r,iroot] += -np.einsum('ik,kjab->ijab',int11,dict_t2[r,iroot]) 
      int11 = 2.0*np.einsum('dclk,ld,ic->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
      dict_Y_ijab[r,iroot] += -np.einsum('ik,kjab->ijab',int11,t2) 
      int11 = 2.0*np.einsum('dclk,ld,ic->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ijab[r,iroot] += -np.einsum('ik,kjab->ijab',int11,t2) 
      int11 = None
                       
                         ##--------------------------------------##
                                 ##diagram non-linear 34'## 
                         ##--------------------------------------##
      
      int12 = -2.0*np.einsum('dclk,ld,ka->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
      dict_Y_ijab[r,iroot] += np.einsum('ca,ijcb->ijab',int12,dict_t2[r,iroot])
      int12 = -2.0*np.einsum('dclk,ld,ka->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
      dict_Y_ijab[r,iroot] += np.einsum('ca,ijcb->ijab',int12,t2)
      int12 = -2.0*np.einsum('dclk,ld,ka->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
      dict_Y_ijab[r,iroot] += np.einsum('ca,ijcb->ijab',int12,t2)
      int12 = None
      ''' 
##----------------------------------------------------------------------------------------------------------##
         #Diagrams and intermediates which include triple excitations (iCCSDn renormalization terms)#
##----------------------------------------------------------------------------------------------------------##

      if (tiCCSD):
        II_oo = intermediates_response.W1_int_So(So)
        II_oo_new = intermediates_response.W1_int_So(dict_So[r,iroot])
        II_vv = intermediates_response.W1_int_Sv(Sv)
        II_vv_new = intermediates_response.W1_int_Sv(dict_Sv[r,iroot])

        II_vo = intermediates_response.coupling_terms_Sv_response(Sv)
        II_vo_new = intermediates_response.coupling_terms_Sv_response(dict_Sv[r,iroot])
        II_ov = intermediates_response.coupling_terms_So_response(So)
        II_ov_new = intermediates_response.coupling_terms_So_response(dict_So[r,iroot])
 
        II_ovoo,II_ovoo3,II_vvvo3 = intermediates_response.w2_int_So_response(So)
        II_ovoo_new,II_ovoo3_new,II_vvvo3_new = intermediates_response.w2_int_So_response(dict_So[r,iroot])
        II_vvvo,II_vvvo2,II_ovoo2 = intermediates_response.w2_int_Sv_response(Sv) 
        II_vvvo_new,II_vvvo2_new,II_ovoo2_new = intermediates_response.w2_int_Sv_response(dict_Sv[r,iroot]) 

##----------------------------------------------------------------------------------------------------------##
                                   #A_lambda So and Sv sector#
##----------------------------------------------------------------------------------------------------------##

        #dict_Y_ia[r,iroot] += amplitude_response.inserted_diag_So_t1(dict_t1[r,iroot],II_oo)
        #dict_Y_ia[r,iroot] += amplitude_response.inserted_diag_So_t1(t1,II_oo_new)
        #dict_Y_ia[r,iroot] += amplitude_response.inserted_diag_Sv_t1(dict_t1[r,iroot],II_vv)
        #dict_Y_ia[r,iroot] += amplitude_response.inserted_diag_Sv_t1(t1,II_vv_new)

        dict_Y_ijab[r,iroot] += amplitude_response.inserted_diag_So(dict_t2[r,iroot],II_oo)   ## (So_T)c terms which simulates triples
        dict_Y_ijab[r,iroot] += amplitude_response.inserted_diag_So(t2,II_oo_new)           ## to renormalize the t2
        dict_Y_ijab[r,iroot] += amplitude_response.inserted_diag_Sv(dict_t2[r,iroot],II_vv)   ## (Sv_T)c terms which simulates triples
        dict_Y_ijab[r,iroot] += amplitude_response.inserted_diag_Sv(t2,II_vv_new)           ## to renormalize the t2
      
      dict_Y_ijab[r,iroot] = cc_symmetrize.symmetrize(dict_Y_ijab[r,iroot])
        
##----------------------------------------------------------------------------------------------------------##
                                     #A_kappa Sv and T sector#
##----------------------------------------------------------------------------------------------------------## 
      
      if (tiCCSD):

        dict_Y_iuab[r,iroot] = amplitude_response.Sv_diagram_vs_contraction_response(dict_Sv[r,iroot]) ##(Fock_Sv)c and (V_Sv)c linear terms contributing to R_iuab

        dict_Y_iuab[r,iroot] += amplitude_response.Sv_diagram_vt_contraction_response(dict_t2[r,iroot]) ##(V_T2)c terms --> R_iuab
        #dict_Y_iuab[r,iroot] += amplitude_response.T1_contribution_Sv_response(dict_t1[r,iroot])       ##(V_T1)c terms --> R_iuab

        #dict_Y_iuab[r,iroot] += amplitude_response.v_so_t_contraction_diag(dict_t2[r,iroot],II_ov) ##(V_So_t)c ----> R_iuab where there is connection b/w So and T
        #dict_Y_iuab[r,iroot] += amplitude_response.v_so_t_contraction_diag(t2,II_ov_new)

        #dict_Y_iuab[r,iroot] += amplitude_response.w2_diag_Sv_response(II_vvvo,II_ovoo3,II_vvvo3,dict_t2[r,iroot])  ##(v_S_t)c -----> R_iuab (diag + off-diag) 
        #dict_Y_iuab[r,iroot] += amplitude_response.w2_diag_Sv_response(II_vvvo_new,II_ovoo3_new,II_vvvo3_new,t2)    ##where both diag and off-diag terms are there also these are with two-body int.

        #dict_Y_iuab[r,iroot] += amplitude_response.nonlinear_Sv_response(II_vv,dict_Sv[r,iroot]) ##Non linear terms of Sv
        #dict_Y_iuab[r,iroot] += amplitude_response.nonlinear_Sv_response(II_vv_new,Sv)
      
##----------------------------------------------------------------------------------------------------------##
                                      #A_kappa So and T sector# 
##----------------------------------------------------------------------------------------------------------##
  
        dict_Y_ijav[r,iroot] = amplitude_response.So_diagram_vs_contraction_response(dict_So[r,iroot])  ##(Fock_So)c and (V_So)c linear terms contributing to R_ijav
        
        dict_Y_ijav[r,iroot] += amplitude_response.So_diagram_vt_contraction_response(dict_t2[r,iroot])  ##(V_T2)c terms --> R_ijav
        #dict_Y_ijav[r,iroot] += amplitude_response.T1_contribution_So_response(dict_t1[r,iroot])         ##(V_T1)c terms --> R_ijav

        #dict_Y_ijav[r,iroot] += amplitude_response.v_sv_t_contraction_diag(dict_t2[r,iroot],II_vo)   ##(V_Sv_t)c ----> R_ijav where there is connection b/w Sv and T
        #dict_Y_ijav[r,iroot] += amplitude_response.v_sv_t_contraction_diag(t2,II_vo_new)    ##one body intermediate and all are off diagonal terms

        #dict_Y_ijav[r,iroot] += amplitude_response.w2_diag_So_response(II_ovoo,II_vvvo2,II_ovoo2,dict_t2[r,iroot]) ##(v_S_t)c -----> R_ijav 
        #dict_Y_ijav[r,iroot] += amplitude_response.w2_diag_So_response(II_ovoo_new,II_vvvo2_new,II_ovoo2_new,t2)  ##where both diag and off-diag terms are there also these are with two-body int.

        #dict_Y_ijav[r,iroot] += amplitude_response.nonlinear_So_response(II_oo,dict_So[r,iroot])     ##Non linear terms of So
        #dict_Y_ijav[r,iroot] += amplitude_response.nonlinear_So_response(II_oo_new,So)
      
##----------------------------------------------------------------------------------------------------------##
                                #Ruling out the linear dependency# 
##----------------------------------------------------------------------------------------------------------##
                 
        for m in range(0,o_act): 
          dict_Y_ijav[r,iroot][:,occ-o_act+m,:,m] = 0.0
          dict_Y_ijav[r,iroot][occ-o_act+m,:,:,m] = 0.0

        for n in range(0,v_act): 
          dict_Y_iuab[r,iroot][:,n,:,n] = 0.0
          dict_Y_iuab[r,iroot][:,n,n,:] = 0.0
        
##----------------------------------------------------------------------------------##
                         #Construction of full B matrix#
##----------------------------------------------------------------------------------##
  
    B_Y_ia_nth = np.zeros((nroot*(r+1),nroot*(r+1)))
    B_Y_ijab_nth = np.zeros((nroot*(r+1),nroot*(r+1)))

    if (tiCCSD):
      B_Y_ijav_nth = np.zeros((nroot*(r+1),nroot*(r+1)))
      B_Y_iuab_nth = np.zeros((nroot*(r+1),nroot*(r+1)))
    
    if (r == 0): 
      B_Y_ia_nth[:nroot,:nroot] = B_Y_ia 
      B_Y_ijab_nth[:nroot,:nroot] = B_Y_ijab

      if (tiCCSD):
        B_Y_ijav_nth[:nroot,:nroot] = B_Y_ijav
        B_Y_iuab_nth[:nroot,:nroot] = B_Y_iuab 
    
    else: 
      B_Y_ia_nth[:nroot*r,:nroot*r] = B_Y_ia 
      B_Y_ijab_nth[:nroot*r,:nroot*r] = B_Y_ijab

      if (tiCCSD):
        B_Y_ijav_nth[:nroot*r,:nroot*r] = B_Y_ijav
        B_Y_iuab_nth[:nroot*r,:nroot*r] = B_Y_iuab 
  
    B_Y_ia = cp.deepcopy(B_Y_ia_nth)
    B_Y_ijab = cp.deepcopy(B_Y_ijab_nth)
    
    if (tiCCSD):
      B_Y_ijav = cp.deepcopy(B_Y_ijav_nth)
      B_Y_iuab = cp.deepcopy(B_Y_iuab_nth)

    B_Y_ia_nth = None
    B_Y_ijab_nth = None

    if (tiCCSD):
      B_Y_ijav_nth = None
      B_Y_iuab_nth = None
  
    for m in range(0,r):
      for iroot in range(0,nroot):
        for jroot in range(0,nroot):
          loc1 = r*nroot+iroot
          loc2 = m*nroot+jroot
  
          B_Y_ia[loc1,loc2] = 2.0*np.einsum('ia,ia',dict_t1[r,iroot],dict_Y_ia[m,jroot])
          B_Y_ia[loc2,loc1] = 2.0*np.einsum('ia,ia',dict_t1[m,jroot],dict_Y_ia[r,iroot])
  
          B_Y_ijab[loc1,loc2] = 2.0*np.einsum('ijab,ijab',dict_t2[r,iroot],dict_Y_ijab[m,jroot])-np.einsum('ijba,ijab',dict_t2[r,iroot],dict_Y_ijab[m,jroot])
          B_Y_ijab[loc2,loc1] = 2.0*np.einsum('ijab,ijab',dict_t2[m,jroot],dict_Y_ijab[r,iroot])-np.einsum('ijba,ijab',dict_t2[m,jroot],dict_Y_ijab[r,iroot])
          
          if (tiCCSD):
            B_Y_ijav[loc1,loc2] = 2.0*np.einsum('ijav,ijav',dict_So[r,iroot],dict_Y_ijav[m,jroot])-np.einsum('jiav,ijav',dict_So[r,iroot],dict_Y_ijav[m,jroot])
            B_Y_ijav[loc2,loc1] = 2.0*np.einsum('ijav,ijav',dict_So[m,jroot],dict_Y_ijav[r,iroot])-np.einsum('jiav,ijav',dict_So[m,jroot],dict_Y_ijav[r,iroot])
  
            B_Y_iuab[loc1,loc2] = 2.0*np.einsum('iuab,iuab',dict_Sv[r,iroot],dict_Y_iuab[m,jroot])-np.einsum('iuba,iuab',dict_Sv[r,iroot],dict_Y_iuab[m,jroot])
            B_Y_iuab[loc2,loc1] = 2.0*np.einsum('iuab,iuab',dict_Sv[m,jroot],dict_Y_iuab[r,iroot])-np.einsum('iuba,iuab',dict_Sv[m,jroot],dict_Y_iuab[r,iroot])
  
    for iroot in range(0,nroot):
      for jroot in range(0,nroot):
        loc1 = r*nroot+iroot
        loc2 = r*nroot+jroot
 
        B_Y_ia[loc1,loc2] = 2.0*np.einsum('ia,ia',dict_t1[r,iroot],dict_Y_ia[r,jroot])
        B_Y_ijab[loc1,loc2] = 2.0*np.einsum('ijab,ijab',dict_t2[r,iroot],dict_Y_ijab[r,jroot])-np.einsum('ijba,ijab',dict_t2[r,iroot],dict_Y_ijab[r,jroot])

        if (tiCCSD):
          B_Y_ijav[loc1,loc2] = 2.0*np.einsum('ijav,ijav',dict_So[r,iroot],dict_Y_ijav[r,jroot])-np.einsum('jiav,ijav',dict_So[r,iroot],dict_Y_ijav[r,jroot])
          B_Y_iuab[loc1,loc2] = 2.0*np.einsum('iuab,iuab',dict_Sv[r,iroot],dict_Y_iuab[r,jroot])-np.einsum('iuba,iuab',dict_Sv[r,iroot],dict_Y_iuab[r,jroot])
     
    B_total = B_Y_ia+B_Y_ijab

    if (tiCCSD):
      B_total += B_Y_ijav+B_Y_iuab

    #print "B Matrix is:", B_total

##-------------------------------------------------------------------------------------##
                          #Diagonalization of the B matrix#
##-------------------------------------------------------------------------------------##

    w_total, vects_total = scipy.linalg.eig(B_total)
    
    
    if (np.all(w_total).imag <= 1e-8):
      w_total = w_total.real    
    #print "Eigenvectors:", vects_total 
    #print "Eigenvalues:", w_total 

##--------------------------------------------------------------------------------------##
                 #Calculation of multiple sorted eigenvalue as#
                #well as the coeff matrix corresponding to that#
##--------------------------------------------------------------------------------------##

    dict_coeff_total = {}
    w = []

##-----------------------------------------------------------------------------------------------------------##
    #for the first iteration the lowest eigen value is chosen but for the subsequent iterations#
    #the desired eigenvalue is guided by the highest overlap of the vectors of a given root with# 
                              #that of the previous iterations#
##-----------------------------------------------------------------------------------------------------------##

##--------------------------------------------------------------------------------------##
       #choosing the lowest eigenvalues and eigen vectors of Zeroth iteration#
##--------------------------------------------------------------------------------------##

    if(r == 0):
      dict_v_nth = {}

      for iroot in range(0,nroot):
        ind_min_wtotal = np.argmin(w_total)                          #to calculate the minimum eigenvalue location from the entire spectrum
        dict_coeff_total[iroot] = vects_total[:,ind_min_wtotal].real  #to calculate the coeff matrix from eigen function corresponding to lowest eigen value
        dict_v_nth[iroot] = dict_coeff_total[iroot]
        w.append(w_total[ind_min_wtotal])
        w_total[ind_min_wtotal] = 123.456

##--------------------------------------------------------------------------------------##
       #choosing the eigen vectors having highest overlap with previous#
                            #iteration vectors#
##--------------------------------------------------------------------------------------##

    if(r>0):
      S_k = {}
      for iroot in range(0,nroot):
        S_k[iroot] = np.zeros((len(w_total)))
        for k in range(0,len(w_total)):
          m = w_total.argsort()[k]
          vect_mth = vects_total[:,m].real
          S_k[iroot][m] = np.abs(np.linalg.multi_dot([dict_v_nth[iroot][:],vect_mth[:r*nroot]]))

        b = np.argmax(S_k[iroot])
        w.append(w_total[b])
        dict_coeff_total[iroot] = vects_total[:,b].real 
        dict_v_nth[iroot] = dict_coeff_total[iroot]
  
##------------------------------------------------------------------------------##
     #Linear Combination of X i.e. t1,t2,So and Sv to form updated vector#
##------------------------------------------------------------------------------##
  
    dict_x_t1  = {}
    dict_x_t2  = {}

    if (tiCCSD):
      dict_x_So  = {}
      dict_x_Sv  = {}
  
    for iroot in range(0,nroot):
      dict_x_t1[r,iroot] = np.zeros((occ,virt))
      dict_x_t2[r,iroot] = np.zeros((occ,occ,virt,virt))    #to calculate the linear combination of X; X' = sum(C_n*x_n)

      if (tiCCSD):
        dict_x_So[r,iroot] = np.zeros((occ,occ,virt,o_act))
        dict_x_Sv[r,iroot] = np.zeros((occ,v_act,virt,virt))
  

      for m in range(0,r+1):
        loc = m*nroot+iroot
        dict_x_t1[r,iroot] += np.linalg.multi_dot([dict_coeff_total[iroot][loc],dict_t1[m,iroot]])
        dict_x_t2[r,iroot] += np.linalg.multi_dot([dict_coeff_total[iroot][loc],dict_t2[m,iroot]])

        if (tiCCSD):
          dict_x_So[r,iroot] += np.linalg.multi_dot([dict_coeff_total[iroot][loc],dict_So[m,iroot]])
          dict_x_Sv[r,iroot] += np.linalg.multi_dot([dict_coeff_total[iroot][loc],dict_Sv[m,iroot]])

      if (tiCCSD):
        lin_norm = davidson.norm_ccsd(dict_x_t1[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot],dict_x_t2[r,iroot])
        #II_int_so = intermediates_response.int_norm_so(dict_x_So[r,iroot],dict_x_So[r,iroot])
        #II_int_sv = intermediates_response.int_norm_sv(dict_x_Sv[r,iroot],dict_x_Sv[r,iroot])
        #lin_norm = davidson.norm_iccsd(dict_x_t1[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot],dict_x_t2[r,iroot],II_int_so, II_int_sv)
        lin_norm = davidson.normalize_iccsd(dict_x_t1[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot],dict_x_t2[r,iroot],dict_x_So[r,iroot],dict_x_So[r,iroot],dict_x_Sv[r,iroot],dict_x_Sv[r,iroot])
      else:
        lin_norm = davidson.norm_ccsd(dict_x_t1[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot],dict_x_t2[r,iroot]) 

      norm = math.sqrt(lin_norm)

      if (norm > 1e-9):
        dict_x_t1[r,iroot] = dict_x_t1[r,iroot]/norm
        dict_x_t2[r,iroot] = dict_x_t2[r,iroot]/norm
      
        if (tiCCSD):
          dict_x_So[r,iroot] = dict_x_So[r,iroot]/norm
          dict_x_Sv[r,iroot] = dict_x_Sv[r,iroot]/norm
      
      lin_norm = davidson.norm_ccsd(dict_x_t1[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot],dict_x_t2[r,iroot])
      #II_int_so = intermediates_response.int_norm_so(dict_x_So[r,iroot],dict_x_So[r,iroot])
      #II_int_sv = intermediates_response.int_norm_sv(dict_x_Sv[r,iroot],dict_x_Sv[r,iroot])
      #lin_norm = davidson.norm_iccsd_temp(dict_x_t1[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot],dict_x_t2[r,iroot],dict_x_So[r,iroot],dict_x_So[r,iroot],dict_x_Sv[r,iroot],dict_x_Sv[r,iroot])
      #lin_norm = davidson.norm_iccsd(dict_x_t1[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot],dict_x_t2[r,iroot],II_int_so, II_int_sv)
      if (tiCCSD):
        lin_norm = davidson.normalize_iccsd(dict_x_t1[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot],dict_x_t2[r,iroot],dict_x_So[r,iroot],dict_x_So[r,iroot],dict_x_Sv[r,iroot],dict_x_Sv[r,iroot])
      else:
        lin_norm = davidson.norm_ccsd(dict_x_t1[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot],dict_x_t2[r,iroot]) 

      print "norm:", lin_norm

##---------------------------------------------------------------------------##
                       #Formation of residual matrix#
##---------------------------------------------------------------------------##
  
    dict_R_ia = {}
    dict_R_ijab = {}

    if (tiCCSD):
      dict_R_ijav = {}
      dict_R_iuab = {}
    
    for iroot in range(0,nroot):                  
      dict_R_ia[r,iroot] = np.zeros((occ,virt))
      dict_R_ijab[r,iroot] = np.zeros((occ,occ,virt,virt))

      if (tiCCSD):                                              #to calculate the residual matrix; (R= AX-wX); R = {Sum(C_n*Y_n)- w*Sum(C_n*x_n)}
        dict_R_ijav[r,iroot] = np.zeros((occ,occ,virt,o_act))   #Where AX is basically the diagram and w is the lowest eigen value. 
        dict_R_iuab[r,iroot] = np.zeros((occ,v_act,virt,virt))
  
      for m in range(0,r+1):
        for jroot in range(0,nroot):
          loc = m*nroot + jroot
          dict_R_ia[r,iroot] += (dict_coeff_total[iroot][loc]*dict_Y_ia[m,jroot] - w[iroot]*dict_coeff_total[iroot][loc]*dict_t1[m,jroot])
          dict_R_ijab[r,iroot] += (dict_coeff_total[iroot][loc]*dict_Y_ijab[m,jroot] - w[iroot]*dict_coeff_total[iroot][loc]*dict_t2[m,jroot])

          if (tiCCSD):
            dict_R_ijav[r,iroot] += (dict_coeff_total[iroot][loc]*dict_Y_ijav[m,jroot] - w[iroot]*dict_coeff_total[iroot][loc]*dict_So[m,jroot])
            dict_R_iuab[r,iroot] += (dict_coeff_total[iroot][loc]*dict_Y_iuab[m,jroot] - w[iroot]*dict_coeff_total[iroot][loc]*dict_Sv[m,jroot])

##----------------------------------------------------------------------------##
                  #Error calculation & convergence criteria#
##----------------------------------------------------------------------------##
  
    eps_t = [] 
  
    if (tiCCSD):
      eps_So = [] 
      eps_Sv = [] 

    for iroot in range(0,nroot):
      eps_t.append(cc_update.update_t1t2(dict_R_ia[r,iroot],dict_R_ijab[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot])[0])
  
      if not (tiCCSD):
        if (eps_t[iroot] <= conv):
          count[iroot] = 1

      if (tiCCSD):

        eps_So.append(cc_update.update_So(dict_R_ijav[r,iroot],dict_x_So[r,iroot])[0])
        eps_Sv.append(cc_update.update_Sv(dict_R_iuab[r,iroot],dict_x_Sv[r,iroot])[0])
        
        if (eps_t[iroot] <= conv and eps_So[iroot] <= conv and eps_Sv[iroot] <= conv):
          count[iroot] = 1

##---------------------------------------------------------------------------##
                       #Convergence and result#
##---------------------------------------------------------------------------##

      print ("             ------------------------")
      if (tiCCSD):
        print 'EPS for IROOT :',iroot+1, '  IS: ', eps_t[iroot], eps_So[iroot], eps_Sv[iroot] 
      else:
        print 'EPS for IROOT :',iroot+1, '  IS: ', eps_t[iroot] 
      print 'Eigenvalue for IROOT :',iroot+1, '  IS:  ', w[iroot], ' a.u. ', w[iroot]*27.2113839, ' eV'
      print ("             ------------------------")


    if (sum(count)== nroot):
      for iroot in range(0,nroot):
        print "!!!!!!!!!!CONVERGED!!!!!!!!!!!!"
        print 'Excitation Energy for sym', isym, 'iroot', iroot+1, ' :', w[iroot], ' a.u. ', w[iroot]*27.2113839, ' eV'

      
        for i in range(0,occ):
          for a in range(0,virt):
            if(abs(dict_x_t1[r,iroot][i,a]) > 5.0*10e-3): 
              print "Highest t1 for root  " ,iroot, " is   ", i,"-->  ", a+occ,"   ", dict_x_t1[r,iroot][i,a] 

        print "---------------------------------"
        for i in range(0,occ):
          for j in range(0,occ):
            for a in range(0,virt):
              for b in range(0,a+1):
                if(abs(dict_x_t2[r,iroot][i,j,a,b]) > 1.0*10e-2): 
                  print "Highest t2 for root  " ,iroot, " is   ", i,"  ",j,"-->  ", a+occ,"  ",b+occ,"   ", dict_x_t2[r,iroot][i,j,a,b] 

        print "---------------------------------"
        if (tiCCSD):
          for i in range(0,occ):
            for u in range(0,v_act):
              for a in range(0,virt):
                for b in range(0,a+1):
                  if(abs(dict_x_Sv[r,iroot][i,u,a,b]) > 1.0*10e-2): 
                    print "Highest Sv for root  " ,iroot, " is   ", i,"  ",u+occ,"-->  ", a+occ,"  ",b+occ,"   ", dict_x_Sv[r,iroot][i,u,a,b] 

          print "---------------------------------"
          for i in range(0,occ):
            for j in range(0,occ):
              for a in range(0,virt):
                for v in range(0,o_act):
                  if(abs(dict_x_So[r,iroot][i,j,a,v]) > 1.0*10e-2): 
                    print "Highest So for root  " ,iroot, " is   ", i,"  ",j,"-->  ", a+occ,"  ",v+occ-o_act,"   ", dict_x_So[r,iroot][i,j,a,v] 
        print "---------------------------------"
        print "---------------------------------"
        print "           "
      break
    
##---------------------------------------------------------------------------##
                        #Calculation of new X#
##---------------------------------------------------------------------------##

    dict_t1_2 = {}
    dict_t2_2 = {}

    if (tiCCSD):
      dict_So_2 = {}
      dict_Sv_2 = {}                     #to get new vector X; (X=R/D-w)

    for iroot in range(0,nroot):
      dict_t1_2[iroot] = davidson.get_XO(dict_R_ia[r,iroot],D1,w[iroot])
      dict_t2_2[iroot] = davidson.get_XO(dict_R_ijab[r,iroot],D2,w[iroot])

      if (tiCCSD):
        dict_So_2[iroot] = davidson.get_XO(dict_R_ijav[r,iroot],Do,w[iroot])
        dict_Sv_2[iroot] = davidson.get_XO(dict_R_iuab[r,iroot],Dv,w[iroot])

##--------------------------------------------------------------------------##
                      #Schmidt orthonormalization#
##--------------------------------------------------------------------------##
    
    dict_ortho_t1 = {} 
    dict_ortho_t2 = {} 
  
    dict_norm_t1 = {} 
    dict_norm_t2 = {} 
  
    if (tiCCSD):
      dict_ortho_So = {} 
      dict_ortho_Sv = {} 

      dict_norm_So = {} 
      dict_norm_Sv = {} 

    for iroot in range(0,nroot):
      dict_ortho_t1[iroot] = dict_t1_2[iroot]
      dict_ortho_t2[iroot] = dict_t2_2[iroot]
 
      if (tiCCSD):
        dict_ortho_So[iroot] = dict_So_2[iroot]
        dict_ortho_Sv[iroot] = dict_Sv_2[iroot]

##---------------------------------------------------------------------------------##
        #Orthonormalization of the vectors corresponding to a given#
               #root with respect to previous iteration vectors#
##---------------------------------------------------------------------------------##

     
      for m in range(0,r+1):
        for jroot in range(0,nroot):
          if (tiCCSD):
            ovrlap = davidson.norm_ccsd(dict_t1_2[iroot],dict_t1[m,jroot],dict_t2_2[iroot],dict_t2[m,jroot])
            #II_ovrlap_so = intermediates_response.int_norm_so(dict_So_2[iroot],dict_So[m,jroot]) 
            #II_ovrlap_sv = intermediates_response.int_norm_sv(dict_Sv_2[iroot],dict_Sv[m,jroot]) 
            #ovrlap = davidson.norm_iccsd(dict_t1_2[iroot],dict_t1[m,jroot],dict_t2_2[iroot],dict_t2[m,jroot],II_ovrlap_so,II_ovrlap_sv)
            ovrlap = davidson.normalize_iccsd(dict_t1_2[iroot],dict_t1[m,jroot],dict_t2_2[iroot],dict_t2[m,jroot],dict_So_2[iroot],dict_So[m,jroot],dict_Sv_2[iroot],dict_Sv[m,jroot])
          else:
            ovrlap = davidson.norm_ccsd(dict_t1_2[iroot],dict_t1[m,jroot],dict_t2_2[iroot],dict_t2[m,jroot])
          
          dict_ortho_t1[iroot] += -ovrlap*dict_t1[m,jroot]    #orthogonalization of same root of different iterations
          dict_ortho_t2[iroot] += -ovrlap*dict_t2[m,jroot]  
       
          if (tiCCSD):
            dict_ortho_So[iroot] += -ovrlap*dict_So[m,jroot]  
            dict_ortho_Sv[iroot] += -ovrlap*dict_Sv[m,jroot]  

##--------------------------------------------------------------------------##
         #Orthonormalization of the vectors of the same iteration# 
##--------------------------------------------------------------------------##

      for jroot in range(0,iroot):
        if (tiCCSD):
         overlap = davidson.norm_ccsd(dict_norm_t1[jroot],dict_t1_2[iroot],dict_norm_t2[jroot],dict_t2_2[iroot])
         #II_overlap_so = intermediates_response.int_norm_so(dict_norm_So[jroot],dict_So_2[iroot])
         #II_overlap_sv = intermediates_response.int_norm_sv(dict_norm_Sv[jroot],dict_Sv_2[iroot])
         #overlap = davidson.norm_iccsd(dict_norm_t1[jroot],dict_t1_2[iroot],dict_norm_t2[jroot],dict_t2_2[iroot],II_overlap_so,II_overlap_sv)
          overlap = davidson.normalize_iccsd(dict_norm_t1[jroot],dict_t1_2[iroot],dict_norm_t2[jroot],dict_t2_2[iroot],dict_norm_So[jroot],dict_So_2[iroot],dict_norm_Sv[jroot],dict_Sv_2[iroot])
        else:
          overlap = davidson.norm_ccsd(dict_norm_t1[jroot],dict_t1_2[iroot],dict_norm_t2[jroot],dict_t2_2[iroot])
         
        dict_ortho_t1[iroot] += -overlap*dict_norm_t1[jroot]   #orthogonalization of different roots of same iteration
        dict_ortho_t2[iroot] += -overlap*dict_norm_t2[jroot]

        if (tiCCSD):
          dict_ortho_So[iroot] += -overlap*dict_norm_So[jroot]  
          dict_ortho_Sv[iroot] += -overlap*dict_norm_Sv[jroot]  

      for m in range(0,r+1):
        for jroot in range(0,nroot):
          if (tiCCSD):
            temp = davidson.norm_ccsd(dict_ortho_t1[iroot],dict_t1[m,jroot],dict_ortho_t2[iroot],dict_t2[m,jroot])
            #II_temp_so = intermediates_response.int_norm_so(dict_ortho_So[iroot],dict_So[m,jroot]) 
            #II_temp_sv = intermediates_response.int_norm_sv(dict_ortho_Sv[iroot],dict_Sv[m,jroot]) 
            #temp = davidson.norm_iccsd(dict_ortho_t1[iroot],dict_t1[m,jroot],dict_ortho_t2[iroot],dict_t2[m,jroot],II_temp_so,II_temp_sv)
            #print "overlap", temp, m , iroot, jroot

      if (tiCCSD):
        ortho_norm = davidson.norm_ccsd(dict_ortho_t1[iroot],dict_ortho_t1[iroot],dict_ortho_t2[iroot],dict_ortho_t2[iroot])
        #II_so = intermediates_response.int_norm_so(dict_ortho_So[iroot],dict_ortho_So[iroot])
        #II_sv = intermediates_response.int_norm_sv(dict_ortho_Sv[iroot],dict_ortho_Sv[iroot])
        #ortho_norm = davidson.norm_iccsd(dict_ortho_t1[iroot],dict_ortho_t1[iroot],dict_ortho_t2[iroot],dict_ortho_t2[iroot],II_so,II_sv)

            temp = davidson.normalize_iccsd(dict_ortho_t1[iroot],dict_t1[m,jroot],dict_ortho_t2[iroot],dict_t2[m,jroot],dict_ortho_So[iroot],dict_So[m,jroot],dict_ortho_Sv[iroot],dict_Sv[m,jroot])
            tmp_t1 = davidson.norm_sep_t1(dict_ortho_t1[iroot],dict_t1[m,jroot])
            tmp_t2 = davidson.norm_sep(dict_ortho_t2[iroot],dict_t2[m,jroot])
            tmp_so = davidson.norm_sep(dict_ortho_So[iroot],dict_So[m,jroot])
            tmp_sv = davidson.norm_sep(dict_ortho_Sv[iroot],dict_Sv[m,jroot])
            print "overlap of t1 vector:", tmp_t1, "t2 vector:", tmp_t2, "So vector:", tmp_so, "Sv vector:", tmp_sv, "& total overlap:", temp, "for roots:", iroot, jroot

      if (tiCCSD):
        ortho_norm = davidson.normalize_iccsd(dict_ortho_t1[iroot],dict_ortho_t1[iroot],dict_ortho_t2[iroot],dict_ortho_t2[iroot],dict_ortho_So[iroot],dict_ortho_So[iroot],dict_ortho_Sv[iroot],dict_ortho_Sv[iroot])
      else:
        ortho_norm = davidson.norm_ccsd(dict_ortho_t1[iroot],dict_ortho_t1[iroot],dict_ortho_t2[iroot],dict_ortho_t2[iroot]) 

      norm_total = math.sqrt(ortho_norm)

      if (norm_total > 1e-12):
        dict_norm_t1[iroot] = dict_ortho_t1[iroot]/norm_total
        dict_norm_t2[iroot] = dict_ortho_t2[iroot]/norm_total

        if (tiCCSD):
          dict_norm_So[iroot] = dict_ortho_So[iroot]/norm_total
          dict_norm_Sv[iroot] = dict_ortho_Sv[iroot]/norm_total

      else:  
        print 'Error in calculation: Generating vector with zero norm'
        quit()
  
      ortho_norm = davidson.norm_ccsd(dict_norm_t1[iroot],dict_norm_t1[iroot],dict_norm_t2[iroot],dict_norm_t2[iroot])
      #II_so = intermediates_response.int_norm_so(dict_norm_So[iroot],dict_norm_So[iroot])
      #II_sv = intermediates_response.int_norm_sv(dict_norm_Sv[iroot],dict_norm_Sv[iroot])
      #ortho_norm = davidson.norm_iccsd(dict_norm_t1[iroot],dict_norm_t1[iroot],dict_norm_t2[iroot],dict_norm_t2[iroot],II_so,II_sv)
      ortho_norm = davidson.normalize_iccsd(dict_norm_t1[iroot],dict_norm_t1[iroot],dict_norm_t2[iroot],dict_norm_t2[iroot],dict_norm_So[iroot],dict_norm_So[iroot],dict_norm_Sv[iroot],dict_norm_Sv[iroot])
      print "norm2:",ortho_norm

##----------------------------------------------------------------------------##
                #updating value of X for the next iteration#
##----------------------------------------------------------------------------##
  
      dict_t1[r+1,iroot] = dict_norm_t1[iroot]
      dict_t2[r+1,iroot] = dict_norm_t2[iroot]

      if (tiCCSD):
         
        for m in range(0,o_act):
          dict_norm_So[iroot][:,occ-o_act+m,:,m] = 0.0
          dict_norm_So[iroot][occ-o_act+m,:,:,m] = 0.0

        for n in range(0,v_act):
          dict_norm_Sv[iroot][:,n,:,n] = 0.0
          dict_norm_Sv[iroot][:,n,n,:] = 0.0
         
        dict_So[r+1,iroot] = dict_norm_So[iroot]
        dict_Sv[r+1,iroot] = dict_norm_Sv[iroot]
 
##----------------------------------------------------------------------------##
                    #Sanity check to find the final norm#
##----------------------------------------------------------------------------##
  
      if (tiCCSD):
        nrm = davidson.norm_ccsd(dict_t1[r+1,iroot],dict_t1[r+1,iroot],dict_t2[r+1,iroot],dict_t2[r+1,iroot])
        #II_norm_so = intermediates_response.int_norm_so(dict_So[r+1,iroot],dict_So[r+1,iroot])
        #II_norm_sv = intermediates_response.int_norm_sv(dict_Sv[r+1,iroot],dict_Sv[r+1,iroot])
        #nrm = davidson.norm_iccsd(dict_t1[r+1,iroot],dict_t1[r+1,iroot],dict_t2[r+1,iroot],dict_t2[r+1,iroot],II_norm_so,II_norm_sv)
        nrm = davidson.normalize_iccsd(dict_t1[r+1,iroot],dict_t1[r+1,iroot],dict_t2[r+1,iroot],dict_t2[r+1,iroot],dict_So[r+1,iroot],dict_So[r+1,iroot],dict_Sv[r+1,iroot],dict_Sv[r+1,iroot])
      else:
        nrm = davidson.norm_ccsd(dict_t1[r+1,iroot],dict_t1[r+1,iroot],dict_t2[r+1,iroot],dict_t2[r+1,iroot]) 

      print "final norm:", iroot, nrm 

##----------------------------------------------------------------------------##
                       #To run the entire routine#
##----------------------------------------------------------------------------##

root_info = inp.nroot 

for i,nroot in enumerate(root_info):
  if (nroot > 0): 
    calc_excitation_energy(i,nroot)
  print 'Done calculation for ', i


                       ##-------------------------------------------------------------------------------------------------------------------------------------------##     
                                                                                           #THE END#
                       ##-------------------------------------------------------------------------------------------------------------------------------------------##     
