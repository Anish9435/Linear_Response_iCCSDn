import numpy as np
import copy as cp
import davidson
import inp
import MP2
import intermediates
import amplitude
import amplitude_response
import cc_symmetrize
import cc_update
import main
import math

##------Import important values-------##

t1 = main.t1
t2 = main.t2
occ = MP2.occ
virt = MP2.virt
nao = MP2.nao
o_act = inp.o_act
v_act = inp.v_act
lrt_iter = inp.lrt_iter
n_davidson = inp.n_davidson
nroot = inp.nroot 
conv = 10**(-inp.conv)
D1 = -1.0*MP2.D1
D2 = -1.0*MP2.D2
dict_t1,dict_t2 = davidson.guess_X(occ,virt,o_act,v_act)
twoelecint_mo = MP2.twoelecint_mo 
count = [0]*nroot

##-----------Initialization of Dictionaries for storing the values----------##

dict_Y_ia = {}
dict_Y_ijab = {}

##-----Initialization of the 1*1 B matrix-------##

B_Y_ia = np.zeros((nroot,nroot))
B_Y_ijab = np.zeros((nroot,nroot))

##------Start iteration--------##
for x in range(0,lrt_iter):

##-----conditioning with the remainder------##

  print ("*********")
  print ("iteration number "+str(x))
  r = x%n_davidson
  print ("Subspace vector "+str(r))
  for iroot in range(0,nroot):
    if(x>0):
      if r==0:
        dict_Y_ia.clear()
        dict_Y_ijab.clear()

        dict_t1.clear()
        dict_t2.clear()
      
        B_Y_ia = np.zeros((nroot*(r+1),nroot*(r+1)))
        B_Y_ijab = np.zeros((nroot*(r+1),nroot*(r+1)))
      
        dict_t1[r,iroot] = dict_x_t1[r,iroot]  
        dict_t2[r,iroot] = dict_x_t1[r,iroot]  
  
##---------------------------------Diagrams and intermediates of CCSD-----------------------------------------------##
  
    I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = intermediates.initialize()

##------------Linear terms of both R_ia and R_ijab---------------------------##

    dict_Y_ia[r,iroot] = amplitude_response.singles_response_linear(I_oo,I_vv,dict_t1[r,iroot],dict_t2[r,iroot])
    dict_Y_ijab[r,iroot] = amplitude_response.doubles_response_linear(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,dict_t1[r,iroot],dict_t2[r,iroot])
##---------------------------------------------------------------------------##  
##----------------update of the intermediates--------------------------------##  
    I1, I2 = intermediates.R_ia_intermediates(t1)
    I1_new, I2_new = intermediates.R_ia_intermediates(dict_t1[r,iroot])
    I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = intermediates.update_int_response(t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)
    I_oo_new,I_vv_new,Ioooo_new,Iovvo_new,Iovvo_2_new,Iovov_new = intermediates.update_int_response(dict_t2[r,iroot],I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)
##-----------------------------------------------------------------------------##  
##----------------Diagrams for R_ia--------------------------------------------##  
    dict_Y_ia[r,iroot] += amplitude_response.singles_response_quadratic(I_oo,I_vv,I1,I2,dict_t1[r,iroot],dict_t2[r,iroot])
    dict_Y_ia[r,iroot] += amplitude_response.singles_response_quadratic(I_oo_new,I_vv_new,I1_new,I2_new,t1,t2)

    dict_Y_ia[r,iroot] += -2*np.einsum('ibkj,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
    dict_Y_ia[r,iroot] +=  2*np.einsum('cbaj,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,dict_t1[r,iroot])
    dict_Y_ia[r,iroot] +=  np.einsum('ibjk,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,dict_t1[r,iroot])
    dict_Y_ia[r,iroot] += -np.einsum('cbja,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1,dict_t1[r,iroot])
  
    dict_Y_ia[r,iroot] += -2*np.einsum('ibkj,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
    dict_Y_ia[r,iroot] +=  2*np.einsum('cbaj,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],dict_t1[r,iroot],t1)
    dict_Y_ia[r,iroot] +=  np.einsum('ibjk,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],dict_t1[r,iroot],t1)
    dict_Y_ia[r,iroot] += -np.einsum('cbja,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],dict_t1[r,iroot],t1)
##--------------Further update of intermediates---------------------------------##  
    I_oo,I_vv,I_oovo,I_vovv,Ioooo_2,I_voov,Iovov_3,Iovvo_3,Iooov,I3 = intermediates.singles_intermediates(t1,t2,I_oo,I_vv,I2)
    I_oo_new,I_vv_new,I_oovo_new,I_vovv_new,Ioooo_2_new,I_voov_new,Iovov_3_new,Iovvo_3_new,Iooov_new,I3_new = intermediates.singles_intermediates(dict_t1[r,iroot],dict_t2[r,iroot],I_oo_new,I_vv_new,I2_new)
##------------------------------------------------------------------------------##
##----------------------Diagrams for R_ijab-------------------------------------##
    dict_Y_ijab[r,iroot] += amplitude_response.doubles_response_quadratic(I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov,dict_t2[r,iroot]) 
    dict_Y_ijab[r,iroot] += amplitude_response.doubles_response_quadratic(I_oo_new,I_vv_new,Ioooo_new,Iovvo_new,Iovvo_2_new,Iovov_new,t2) 
    dict_Y_ijab[r,iroot] += amplitude_response.singles_n_doubles_response(dict_t1[r,iroot],I_oovo,I_vovv)
    dict_Y_ijab[r,iroot] += amplitude_response.singles_n_doubles_response(t1,I_oovo_new,I_vovv_new)

    dict_Y_ijab[r,iroot] += -np.einsum('ickb,ka,jc->ijab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,dict_t1[r,iroot])   #diagrams non-linear 3
    dict_Y_ijab[r,iroot] += -np.einsum('icak,jc,kb->ijab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,dict_t1[r,iroot])   #diagrams non-linear 4
    dict_Y_ijab[r,iroot] += 0.5*np.einsum('ijkl,ka,lb->ijab',twoelecint_mo[:occ,:occ,:occ,:occ],t1,dict_t1[r,iroot])      #diagram non-linear 1
    dict_Y_ijab[r,iroot] += 0.5*np.einsum('cdab,ic,jd->ijab',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],t1,dict_t1[r,iroot])      #diagram non-linear 2
  
    dict_Y_ijab[r,iroot] += -np.einsum('ickb,ka,jc->ijab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],dict_t1[r,iroot],t1)   #diagrams non-linear 3
    dict_Y_ijab[r,iroot] += -np.einsum('icak,jc,kb->ijab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],dict_t1[r,iroot],t1)   #diagrams non-linear 4
    dict_Y_ijab[r,iroot] += 0.5*np.einsum('ijkl,ka,lb->ijab',twoelecint_mo[:occ,:occ,:occ,:occ],dict_t1[r,iroot],t1)      #diagram non-linear 1
    dict_Y_ijab[r,iroot] += 0.5*np.einsum('cdab,ic,jd->ijab',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],dict_t1[r,iroot],t1)      #diagram non-linear 2
    dict_Y_ijab[r,iroot] = cc_symmetrize.symmetrize(dict_Y_ijab[r,iroot])
##------------------------------------------------------------------------------##

##---------Construction of B matrix---------------##

  B_Y_ia_nth = np.zeros((nroot*(r+1),nroot*(r+1)))
  B_Y_ijab_nth = np.zeros((nroot*(r+1),nroot*(r+1)))

  if (r==0):
    B_Y_ia_nth[:nroot,:nroot] = B_Y_ia
    B_Y_ijab_nth[:nroot,:nroot] = B_Y_ijab
  else:
    B_Y_ia_nth[:nroot*r,:nroot*r] = B_Y_ia 
    B_Y_ijab_nth[:nroot*r,:nroot*r] = B_Y_ijab 
  
  B_Y_ia = cp.deepcopy(B_Y_ia_nth)
  B_Y_ijab = cp.deepcopy(B_Y_ijab_nth)
    
  B_Y_ia_nth = None
  B_Y_ijab_nth = None

  for m in range(0,r+1):
    for iroot in range(0,nroot):
      for jroot in range(0,nroot):
        loc1 = r*nroot+iroot
        loc2 = m*nroot+jroot
        B_Y_ia[loc2,loc1] = 2.0*np.einsum('ia,ia',dict_t1[m,iroot],dict_Y_ia[r,jroot])
        B_Y_ijab[loc2,loc1] = 2.0*np.einsum('ijab,ijab',dict_t2[m,iroot],dict_Y_ijab[r,jroot])-np.einsum('ijba,ijab',dict_t2[m,iroot],dict_Y_ijab[r,jroot])

        if (r!=m):
          B_Y_ia[loc1,loc2] = 2.0*np.einsum('ia,ia',dict_t1[r,jroot],dict_Y_ia[m,iroot])
          B_Y_ijab[loc1,loc2] = 2.0*np.einsum('ijab,ijab',dict_t2[r,jroot],dict_Y_ijab[m,iroot])-np.einsum('ijba,ijab',dict_t2[r,jroot],dict_Y_ijab[m,iroot])
  B_total = B_Y_ia+B_Y_ijab
  #print B_total

##-------Diagonalization of the B matrix----------##
      
  w_total, vects_total = np.linalg.eig(B_total)
  
  if (np.all(w_total).imag <= 1e-8):
    w_total = w_total.real    
  #print vects_total
##-----Assigning the position of the lowest eigenvalue----##

  dict_coeff_total = {}
  w = []
  for iroot in range(0,nroot):
    ind_min_wtotal = np.argmin(w_total)
    dict_coeff_total[iroot] = vects_total[:,ind_min_wtotal]
    w.append(w_total[ind_min_wtotal])
    w_total[ind_min_wtotal] = 123.456

##-----------------Linear Combination--------------##

  dict_x_t1  = {}
  dict_x_t2  = {}

  for iroot in range(0,nroot):
    dict_x_t1[r,iroot] = np.zeros((occ,virt))
    dict_x_t2[r,iroot] = np.zeros((occ,occ,virt,virt))
    for m in range(0,r+1):
      for jroot in range(0,nroot):
        loc = m*nroot+jroot
        dict_x_t1[r,iroot] += np.linalg.multi_dot([dict_coeff_total[iroot][loc],dict_t1[m,jroot]])
        dict_x_t2[r,iroot] += np.linalg.multi_dot([dict_coeff_total[iroot][loc],dict_t2[m,jroot]])

        lin_norm = 2.0*np.einsum('ia,ia',dict_x_t1[r,iroot],dict_x_t1[r,iroot])
        lin_norm += 2.0*np.einsum('ijab,ijab',dict_x_t2[r,iroot],dict_x_t2[r,iroot])-np.einsum('ijab,ijba',dict_x_t2[r,iroot],dict_x_t2[r,iroot])
    norm = math.sqrt(lin_norm)
 
    if (norm > 1e-9):
      dict_x_t1[r,iroot] = dict_x_t1[r,iroot]/norm
      dict_x_t2[r,iroot] = dict_x_t2[r,iroot]/norm
    #print dict_x_t1
 
##------Formation of residual matrix--------##
  
  dict_R_ia = {}
  dict_R_ijab = {}
  
  for iroot in range(0,nroot):
    dict_R_ia[r,iroot] = np.zeros((occ,virt))
    dict_R_ijab[r,iroot] = np.zeros((occ,occ,virt,virt))
    for m in range(0,r+1):
      for jroot in range(0,nroot):
        loc = m*nroot + jroot
        dict_R_ia[r,iroot] += (dict_coeff_total[iroot][loc]*dict_Y_ia[m,jroot] - w[iroot]*dict_coeff_total[iroot][loc]*dict_t1[m,jroot])
        dict_R_ijab[r,iroot] += (dict_coeff_total[iroot][loc]*dict_Y_ijab[m,jroot] - w[iroot]*dict_coeff_total[iroot][loc]*dict_t2[m,jroot])
  
  eps_t = [] 
  for iroot in range(0,nroot):
    eps_t.append(cc_update.update_t1t2(dict_R_ia[r,iroot],dict_R_ijab[r,iroot],dict_x_t1[r,iroot],dict_x_t2[r,iroot])[0])
  
    print 'EPS: ', eps_t[iroot] 
    print 'Eigen value: ', w[iroot]
    if (eps_t[iroot] <= conv):
      count[iroot] = 1
  if (sum(count)== nroot):
    print "CONVERGED!!!!!!!!!!!!"
    print 'Excitation Energy: ', w[iroot]
    break

##--------Divide residue by denominator to get X--------##

  dict_t1_2 = {}
  dict_t2_2 = {}

  for iroot in range(0,nroot):
    dict_t1_2[iroot] = davidson.get_XO(dict_R_ia[r,iroot],D1,w[iroot])
    dict_t2_2[iroot] = davidson.get_XO(dict_R_ijab[r,iroot],D2,w[iroot])

##------Schmidt orthogonalization----------##
 
  dict_ortho_t1 = {} 
  dict_ortho_t2 = {} 

  dict_norm_t1 = {} 
  dict_norm_t2 = {} 

  for iroot in range(0,nroot):
    dict_ortho_t1[iroot] = dict_t1_2[iroot]
    dict_ortho_t2[iroot] = dict_t2_2[iroot]
  
    for m in range(0,r+1):
      ovrlap = 2.0*np.einsum('ia,ia',dict_t1_2[iroot],dict_t1[m,iroot]) + 2.0*np.einsum('ijab,ijab',dict_t2_2[iroot],dict_t2[m,iroot]) - np.einsum('ijab,ijba',dict_t2_2[iroot],dict_t2[m,iroot])
      dict_ortho_t1[iroot] += - ovrlap*dict_t1[m,iroot]  
      dict_ortho_t2[iroot] += - ovrlap*dict_t2[m,iroot]  
  
      for jroot in range(0,iroot):
        overlap = 2.0*np.einsum('ia,ia',dict_norm_t1[jroot],dict_t1_2[iroot]) + 2.0*np.einsum('ijab,ijab',dict_norm_t2[jroot],dict_t2_2[iroot]) - np.einsum('ijab,ijba',dict_norm_t2[jroot],dict_t2_2[iroot])
        dict_ortho_t1[iroot] += -overlap*dict_norm_t1[jroot]
        dict_ortho_t2[iroot] += -overlap*dict_norm_t2[jroot]
    #print dict_ortho_t1 

##--------Normalization of the new t and s------##

    ortho_norm = 2.0*np.einsum('ia,ia',dict_ortho_t1[iroot],dict_ortho_t1[iroot])
    ortho_norm += 2.0*np.einsum('ijab,ijab',dict_ortho_t2[iroot],dict_ortho_t2[iroot])-np.einsum('ijab,ijba',dict_ortho_t2[iroot],dict_ortho_t2[iroot])
    norm_total = math.sqrt(ortho_norm)

  # Each part of the vector is divided by the total norm of the vector
    if (norm_total > 1e-9):
      dict_norm_t1[iroot] = dict_ortho_t1[iroot]/norm_total
      dict_norm_t2[iroot] = dict_ortho_t2[iroot]/norm_total
    else:  
      print 'Error in calculation: Generating vector with zero norm'
      quit()
  
##-------updating value of X for the next iteration-------##

    dict_t1[r+1,iroot] = dict_norm_t1[iroot]
    dict_t2[r+1,iroot] = dict_norm_t2[iroot]
    nrm = 2.0*np.einsum('ia,ia',dict_t1[r+1,iroot],dict_t1[r+1,iroot]) + 2.0*np.einsum('ijab,ijab',dict_t2[r+1,iroot],dict_t2[r+1,iroot]) - np.einsum('ijab,ijba',dict_t2[r+1,iroot],dict_t2[r+1,iroot])
    print "final norm:", iroot, nrm
    #print dict_t1[r+1,iroot]

  #for iroot in range(0,nroot):
  #  for m in range(0,r+1):
  #    p = 2.0*np.einsum('ia,ia',dict_t1[m,iroot],dict_ortho_t1[iroot])
  #    q = 2.0*np.einsum('ijab,ijab',dict_t2[m,iroot],dict_ortho_t2[iroot])-np.einsum('ijab,ijba',dict_t2[m,iroot],dict_ortho_t2[iroot])
  #    y = p+q
  #    print "overlap:", iroot,p,q,y  
