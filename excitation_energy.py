                   
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
D1 = 1.0*MP2.D1
D2 = 1.0*MP2.D2

if (tiCCSD):
  So = main.So
  Sv = main.Sv
  Do = 1.0*MP2.Do 
  Dv = 1.0*MP2.Dv 

conv = 1*(10**(-inp.LR_conv))

sym = trans_mo.mol.symmetry
orb_sym_pyscf = trans_mo.orb_symm
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
  
##----------------------------------------------------------------------------------------------------------##
         #Diagrams and intermediates which include triple excitations (iCCSDn renormalization terms)#
##----------------------------------------------------------------------------------------------------------##

      if (tiCCSD):
        II_oo = intermediates_response.W1_int_So(So)
        II_oo_new = intermediates_response.W1_int_So(dict_So[r,iroot])
        II_vv = intermediates_response.W1_int_Sv(Sv)
        II_vv_new = intermediates_response.W1_int_Sv(dict_Sv[r,iroot])
  
##----------------------------------------------------------------------------------------------------------##
                                   #A_lambda So and Sv sector#
##----------------------------------------------------------------------------------------------------------##

        #dict_Y_ia[r,iroot] += amplitude_response.inserted_diag_So_t1(dict_t1[r,iroot],II_oo)
        #dict_Y_ia[r,iroot] += amplitude_response.inserted_diag_So_t1(t1,II_oo_new)
        #dict_Y_ia[r,iroot] += amplitude_response.inserted_diag_Sv_t1(dict_t1[r,iroot],II_vv)
        #dict_Y_ia[r,iroot] += amplitude_response.inserted_diag_Sv_t1(t1,II_vv_new)

        dict_Y_ijab[r,iroot] += amplitude_response.inserted_diag_So(dict_t2[r,iroot],II_oo)
        dict_Y_ijab[r,iroot] += amplitude_response.inserted_diag_So(t2,II_oo_new)
        dict_Y_ijab[r,iroot] += amplitude_response.inserted_diag_Sv(dict_t2[r,iroot],II_vv)
        dict_Y_ijab[r,iroot] += amplitude_response.inserted_diag_Sv(t2,II_vv_new)

      dict_Y_ijab[r,iroot] = cc_symmetrize.symmetrize(dict_Y_ijab[r,iroot])
        
##----------------------------------------------------------------------------------------------------------##
                                     #A_kappa Sv and T sector#
##----------------------------------------------------------------------------------------------------------## 
      
      if (tiCCSD):
        dict_Y_iuab[r,iroot] = amplitude_response.Sv_diagram_vs_contraction_response(dict_Sv[r,iroot])

        dict_Y_iuab[r,iroot] += amplitude_response.Sv_diagram_vt_contraction_response(dict_t2[r,iroot])
        dict_Y_iuab[r,iroot] += amplitude_response.T1_contribution_Sv_response(dict_t1[r,iroot])
  
##----------------------------------------------------------------------------------------------------------##
                                      #A_kappa So and T sector# 
##----------------------------------------------------------------------------------------------------------##
  
        dict_Y_ijav[r,iroot] = amplitude_response.So_diagram_vs_contraction_response(dict_So[r,iroot])

        dict_Y_ijav[r,iroot] += amplitude_response.So_diagram_vt_contraction_response(dict_t2[r,iroot])
        dict_Y_ijav[r,iroot] += amplitude_response.T1_contribution_So_response(dict_t1[r,iroot])
    

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

##-------------------------------------------------------------------------------------##
                          #Diagonalization of the B matrix#
##-------------------------------------------------------------------------------------##

    w_total, vects_total = np.linalg.eig(B_total)
    
    
    if (np.all(w_total).imag <= 1e-8):
      w_total = w_total.real    
    print w_total

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
        for jroot in range(0,nroot):
          loc = m*nroot+jroot
          dict_x_t1[r,iroot] += np.linalg.multi_dot([dict_coeff_total[iroot][loc],dict_t1[m,jroot]])
          dict_x_t2[r,iroot] += np.linalg.multi_dot([dict_coeff_total[iroot][loc],dict_t2[m,jroot]])

          if (tiCCSD):
            dict_x_So[r,iroot] += np.linalg.multi_dot([dict_coeff_total[iroot][loc],dict_So[m,jroot]])
            dict_x_Sv[r,iroot] += np.linalg.multi_dot([dict_coeff_total[iroot][loc],dict_Sv[m,jroot]])

      lin_norm = 2.0*np.einsum('ia,ia',dict_x_t1[r,iroot],dict_x_t1[r,iroot])
      lin_norm += 2.0*np.einsum('ijab,ijab',dict_x_t2[r,iroot],dict_x_t2[r,iroot]) - np.einsum('ijab,ijba',dict_x_t2[r,iroot],dict_x_t2[r,iroot])

      if (tiCCSD):
        lin_norm += 2.0*np.einsum('ijav,ijav',dict_x_So[r,iroot],dict_x_So[r,iroot]) - np.einsum('ijav,jiav',dict_x_So[r,iroot],dict_x_So[r,iroot])
        lin_norm += 2.0*np.einsum('iuab,iuab',dict_x_Sv[r,iroot],dict_x_Sv[r,iroot]) - np.einsum('iuab,iuba',dict_x_Sv[r,iroot],dict_x_Sv[r,iroot])

      norm = math.sqrt(lin_norm)

      if (norm > 1e-9):
        dict_x_t1[r,iroot] = dict_x_t1[r,iroot]/norm
        dict_x_t2[r,iroot] = dict_x_t2[r,iroot]/norm
      
        if (tiCCSD):
          dict_x_So[r,iroot] = dict_x_So[r,iroot]/norm
          dict_x_Sv[r,iroot] = dict_x_Sv[r,iroot]/norm
      
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
          ovrlap = 2.0*np.einsum('ia,ia',dict_t1_2[iroot],dict_t1[m,jroot]) 
          ovrlap += 2.0*np.einsum('ijab,ijab',dict_t2_2[iroot],dict_t2[m,jroot]) - np.einsum('ijab,ijba',dict_t2_2[iroot],dict_t2[m,jroot]) 

          if (tiCCSD):
            ovrlap += 2.0*np.einsum('ijav,ijav',dict_So_2[iroot],dict_So[m,jroot]) - np.einsum('ijav,jiav',dict_So_2[iroot],dict_So[m,jroot]) 
            ovrlap += 2.0*np.einsum('iuab,iuab',dict_Sv_2[iroot],dict_Sv[m,jroot]) - np.einsum('iuab,iuba',dict_Sv_2[iroot],dict_Sv[m,jroot])
       
          dict_ortho_t1[iroot] += -ovrlap*dict_t1[m,jroot]    #orthogonalization of same root of different iterations
          dict_ortho_t2[iroot] += -ovrlap*dict_t2[m,jroot]  
       
          if (tiCCSD):
            dict_ortho_So[iroot] += -ovrlap*dict_So[m,jroot]  
            dict_ortho_Sv[iroot] += -ovrlap*dict_Sv[m,jroot]  

##--------------------------------------------------------------------------##
         #Orthonormalization of the vectors of the same iteration# 
##--------------------------------------------------------------------------##

      for jroot in range(0,iroot):
        overlap = 2.0*np.einsum('ia,ia',dict_norm_t1[jroot],dict_t1_2[iroot]) 
        overlap += 2.0*np.einsum('ijab,ijab',dict_norm_t2[jroot],dict_t2_2[iroot]) - np.einsum('ijab,ijba',dict_norm_t2[jroot],dict_t2_2[iroot]) 

        if (tiCCSD):
          overlap += 2.0*np.einsum('ijav,ijav',dict_norm_So[jroot],dict_So_2[iroot]) - np.einsum('ijav,jiav',dict_norm_So[jroot],dict_So_2[iroot]) 
          overlap += 2.0*np.einsum('iuab,iuab',dict_norm_Sv[jroot],dict_Sv_2[iroot]) - np.einsum('iuab,iuba',dict_norm_Sv[jroot],dict_Sv_2[iroot])

        dict_ortho_t1[iroot] += -overlap*dict_norm_t1[jroot]   #orthogonalization of different roots of same iteration
        dict_ortho_t2[iroot] += -overlap*dict_norm_t2[jroot]

        if (tiCCSD):
          dict_ortho_So[iroot] += -overlap*dict_norm_So[jroot]  
          dict_ortho_Sv[iroot] += -overlap*dict_norm_Sv[jroot]  


      ortho_norm = 2.0*np.einsum('ia,ia',dict_ortho_t1[iroot],dict_ortho_t1[iroot])
      ortho_norm += 2.0*np.einsum('ijab,ijab',dict_ortho_t2[iroot],dict_ortho_t2[iroot]) - np.einsum('ijab,ijba',dict_ortho_t2[iroot],dict_ortho_t2[iroot])

      if (tiCCSD):
        ortho_norm += 2.0*np.einsum('ijav,ijav',dict_ortho_So[iroot],dict_ortho_So[iroot]) - np.einsum('ijav,jiav',dict_ortho_So[iroot],dict_ortho_So[iroot])
        ortho_norm += 2.0*np.einsum('iuab,iuab',dict_ortho_Sv[iroot],dict_ortho_Sv[iroot]) - np.einsum('iuab,iuba',dict_ortho_Sv[iroot],dict_ortho_Sv[iroot])

      norm_total = math.sqrt(ortho_norm)

      if (norm_total > 1e-9):
        dict_norm_t1[iroot] = dict_ortho_t1[iroot]/norm_total
        dict_norm_t2[iroot] = dict_ortho_t2[iroot]/norm_total

        if (tiCCSD):
          dict_norm_So[iroot] = dict_ortho_So[iroot]/norm_total
          dict_norm_Sv[iroot] = dict_ortho_Sv[iroot]/norm_total

      else:  
        print 'Error in calculation: Generating vector with zero norm'
        quit()
  
##----------------------------------------------------------------------------##
                #updating value of X for the next iteration#
##----------------------------------------------------------------------------##
  
      dict_t1[r+1,iroot] = dict_norm_t1[iroot]
      dict_t2[r+1,iroot] = dict_norm_t2[iroot]

      if (tiCCSD):
        dict_So[r+1,iroot] = dict_norm_So[iroot]
        dict_Sv[r+1,iroot] = dict_norm_Sv[iroot]
 
##----------------------------------------------------------------------------##
                    #Sanity check to find the final norm#
##----------------------------------------------------------------------------##
  
      nrm = 2.0*np.einsum('ia,ia',dict_t1[r+1,iroot],dict_t1[r+1,iroot]) 
      nrm += 2.0*np.einsum('ijab,ijab',dict_t2[r+1,iroot],dict_t2[r+1,iroot]) - np.einsum('ijab,ijba',dict_t2[r+1,iroot],dict_t2[r+1,iroot]) 

      if (tiCCSD):
        nrm += 2.0*np.einsum('ijav,ijav',dict_So[r+1,iroot],dict_So[r+1,iroot]) - np.einsum('ijav,jiav',dict_So[r+1,iroot],dict_So[r+1,iroot]) 
        nrm += 2.0*np.einsum('iuab,iuab',dict_Sv[r+1,iroot],dict_Sv[r+1,iroot]) - np.einsum('iuab,iuba',dict_Sv[r+1,iroot],dict_Sv[r+1,iroot])

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
