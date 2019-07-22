import numpy as np
import copy as cp
import davidson
import inp
import MP2
import intermediates
import amplitude
import cc_symmetrize
import cc_update
import main

##------Import important values-------##

t1=main.t1
t2=main.t2
So=main.So
Sv=main.Sv
occ = MP2.occ
virt = MP2.virt
o_act = inp.o_act
v_act = inp.v_act
n_iter = inp.n_iter
n_davidson = inp.n_davidson
conv = 10**(-inp.conv)
D1 = MP2.D1
D2 = MP2.D2
Do = MP2.Do
Dv = MP2.Dv
t1_new,t2_new,So_new,Sv_new = davidson.guess_X(occ,virt,o_act,v_act)

##-----------Initialization of Dictionaries for storing the values----------##

dict_Y_ia = {}
dict_Y_ijab = {}
dict_Y_ijav = {}
dict_Y_iuab = {}

dict_t1 = {}
dict_t2 = {}
dict_So = {}
dict_Sv = {}

##-----Storing the guess vectors in the directory-----##

dict_t1[0] = t1_new
dict_t2[0] = t2_new
dict_So[0] = So_new
dict_Sv[0] = Sv_new

##-----Initialization of the 1*1 B matrix-------##

B_Y_ia = np.zeros((1,1))
B_Y_ijab = np.zeros((1,1))
B_Y_ijav = np.zeros((1,1))
B_Y_iuab = np.zeros((1,1))

##------Start iteration--------##
for x in range(0,n_iter):

##-----conditioning with the remainder------##

  print ("iteration number"+str(x))
  r = x%n_davidson
  if(x>0):  
    if r==0:
      dict_Y_ia.clear()
      dict_Y_ijab.clear()
      dict_Y_ijav.clear()
      dict_Y_iuab.clear()

      dict_t1.clear()
      dict_t2.clear()
      dict_So.clear()
      dict_Sv.clear()
      
      B_Y_ia = np.zeros((r+1,r+1))
      B_Y_ijab = np.zeros((r+1,r+1))
      B_Y_ijav = np.zeros((r+1,r+1))
      B_Y_iuab = np.zeros((r+1,r+1))
      '''
      dict_t1[r] = ortho_t1
      dict_t2[r] = ortho_t2
      dict_So[r] = ortho_So
      dict_Sv[r] = ortho_Sv
      '''
      dict_t1[r] = norm_t1
      dict_t2[r] = norm_t2
      dict_So[r] = norm_So
      dict_Sv[r] = norm_Sv
  
##------Diagrams of coupled cluster theory i.e AX with new t and s---------##
  
  tau = cp.deepcopy(dict_t2[r]) 
  I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2, I_oovo, I_vovv = intermediates.initialize()
  I1, I2 = intermediates.R_ia_intermediates(t1)
  II_oo = intermediates.W1_int_So(So)
  II_vv = intermediates.W1_int_Sv(Sv)
  II_oo_new = intermediates.W1_int_So(dict_So[r])
  II_vv_new = intermediates.W1_int_Sv(dict_Sv[r])
  
  Y_ia = amplitude.singles(I1,I2,I_oo,I_vv,tau,dict_t1[r],dict_t2[r])
  Y_ijab = amplitude.doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,dict_t2[r])
  Y_ijab += amplitude.singles_n_doubles(dict_t1[r],dict_t2[r],tau,I_oovo,I_vovv) 
 
  Y_ijab += amplitude.inserted_diag_So(dict_t2[r],II_oo) 
  Y_ijab += amplitude.inserted_diag_So(t2,II_oo_new) 
  Y_ijab += amplitude.inserted_diag_Sv(dict_t2[r],II_vv)
  Y_ijab += amplitude.inserted_diag_Sv(t2,II_vv_new)
  Y_ijab = cc_symmetrize.symmetrize(Y_ijab)
 
  Y_iuab = amplitude.Sv_diagram_vs_contraction(dict_Sv[r],II_vv_new)
  Y_iuab += amplitude.Sv_diagram_vt_contraction(dict_t2[r])
  Y_ijav = amplitude.So_diagram_vs_contraction(dict_So[r],II_oo_new)
  Y_ijav += amplitude.So_diagram_vt_contraction(dict_t2[r])
  
##-------Storing AX in the dictionary---------##
      
  dict_Y_ia[r] = Y_ia
  dict_Y_ijab[r] = Y_ijab
  dict_Y_ijav[r] = Y_ijav
  dict_Y_iuab[r] = Y_iuab

##---------Construction of B matrix---------------##

  #print "----B Matrix-------" 
  B_Y_ia_nth = np.zeros((r+1,r+1))
  B_Y_ijab_nth = np.zeros((r+1,r+1))
  B_Y_ijav_nth = np.zeros((r+1,r+1))
  B_Y_iuab_nth = np.zeros((r+1,r+1))
  
  B_Y_ia_nth[:r,:r] = B_Y_ia 
  B_Y_ijab_nth[:r,:r] = B_Y_ijab
  B_Y_ijav_nth[:r,:r] = B_Y_ijav
  B_Y_iuab_nth[:r,:r] = B_Y_iuab 
  
  B_Y_ia = cp.deepcopy(B_Y_ia_nth)
  B_Y_ijab = cp.deepcopy(B_Y_ijab_nth)
  B_Y_ijav = cp.deepcopy(B_Y_ijav_nth)
  B_Y_iuab = cp.deepcopy(B_Y_iuab_nth)
  
  B_Y_ia_nth = None
  B_Y_ijab_nth = None
  B_Y_ijav_nth = None
  B_Y_iuab_nth = None
 

  for m in range(1,r+1):
    B_Y_ia[m-1,r] = np.einsum('ia,ia',dict_t1[m-1],dict_Y_ia[r])
    B_Y_ia[r,m-1] = np.einsum('ia,ia',dict_t1[r],dict_Y_ia[m-1])
    B_Y_ijab[m-1,r] = np.einsum('ijab,ijab',dict_t2[m-1],dict_Y_ijab[r])
    B_Y_ijab[r,m-1] = np.einsum('ijab,ijab',dict_t2[r],dict_Y_ijab[m-1])
    B_Y_ijav[m-1,r] = np.einsum('ijav,ijav',dict_So[m-1],dict_Y_ijav[r])
    B_Y_ijav[r,m-1] = np.einsum('ijav,ijav',dict_So[r],dict_Y_ijav[m-1])
    B_Y_iuab[m-1,r] = np.einsum('iuab,iuab',dict_Sv[m-1],dict_Y_iuab[r])
    B_Y_iuab[r,m-1] = np.einsum('iuab,iuab',dict_Sv[r],dict_Y_iuab[m-1])

  B_Y_ia[r,r] = np.einsum('ia,ia',dict_t1[r],dict_Y_ia[r])
  B_Y_ijab[r,r] = np.einsum('ijab,ijab',dict_t2[r],dict_Y_ijab[r])
  B_Y_ijav[r,r] = np.einsum('ijav,ijav',dict_So[r],dict_Y_ijav[r])
  B_Y_iuab[r,r] = np.einsum('iuab,iuab',dict_Sv[r],dict_Y_iuab[r])
   
  B_total = B_Y_ia+B_Y_ijab+B_Y_iuab+B_Y_ijav
  print B_total

##-------Diagonalization of the B matrix----------##
      
  w_total, vects_total = np.linalg.eig(B_total)
  print w_total
  print vects_total
 
##--Assigning the position of the lowest eigenvalue----##

  ind_min_wtotal = np.argmin(w_total)
##------Coefficient matrix corresponding to lowest eigenvalue-----##
    
  coeff_total = vects_total[:,ind_min_wtotal]
  print coeff_total

##------Linear Combination--------------##

  x_t1 = np.zeros((occ,virt))
  x_t2 = np.zeros((occ,occ,virt,virt))      ###x_t1 and so on are the linear combination of the coefficient vector elements and the t1,t1,So and Sv values of successive iterations###
  x_So = np.zeros((occ,occ,virt,o_act))
  x_Sv = np.zeros((occ,v_act,virt,virt))

  for i in range(0,r+1):
    x_t1 += np.linalg.multi_dot([coeff_total[i],dict_t1[i]])
    x_t2 += np.linalg.multi_dot([coeff_total[i],dict_t2[i]])
    x_So += np.linalg.multi_dot([coeff_total[i],dict_So[i]])
    x_Sv += np.linalg.multi_dot([coeff_total[i],dict_Sv[i]])

##------Formation of residual matrix--------##
  
  R_ia = np.zeros((occ,virt))
  R_ijab = np.zeros((occ,occ,virt,virt))      
  R_ijav = np.zeros((occ,occ,virt,o_act))
  R_iuab = np.zeros((occ,v_act,virt,virt))

  for i in range(0,r+1): 
    R_ia += (coeff_total[i]*dict_Y_ia[i] - w_total[ind_min_wtotal]*x_t1)
    R_ijab += (coeff_total[i]*dict_Y_ijab[i] - w_total[ind_min_wtotal]*x_t2)
    R_ijav += (coeff_total[i]*dict_Y_ijav[i] - w_total[ind_min_wtotal]*x_So)
    R_iuab += (coeff_total[i]*dict_Y_iuab[i] - w_total[ind_min_wtotal]*x_Sv)

  eps_t = cc_update.update_t1t2(R_ia,R_ijab,x_t1,x_t2)[0]
  eps_So = cc_update.update_So(R_ijav,x_So)[0]
  eps_Sv = cc_update.update_Sv(R_iuab,x_Sv)[0]
  
  if (eps_t <= conv and eps_So <= conv and eps_Sv <= conv):
    print "CONVERGED!!!!!!!!!!!!"
    break

##--------Divide residue by denominator to get X--------##

  t1_2 = davidson.get_X(R_ia,D1)
  t2_2 = davidson.get_X(R_ijab,D2)
  So_2 = davidson.get_X(R_ijav,Do)
  Sv_2 = davidson.get_X(R_iuab,Dv)

##------Schmidt orthogonalization----------##
  ortho_t1 = t1_2 
  ortho_t2 = t2_2 
  ortho_So = So_2 
  ortho_Sv = Sv_2
  ''' 
  for i in range(0,r+1):
    ortho_t1 += - np.einsum('ia,ia',dict_t1[i],norm_t1)*dict_t1[i]
    ortho_t2 += - np.einsum('ijab,ijab',dict_t2[i],norm_t2)*dict_t2[i]
    ortho_So += - np.einsum('ijav,ijav',dict_So[i],norm_So)*dict_So[i]
    ortho_Sv += - np.einsum('iuab,iuab',dict_Sv[i],norm_Sv)*dict_Sv[i]
  ''' 
  for i in range(0,r+1):
    ortho_t1 += - np.einsum('ia,ia',dict_t1[i],t1_2)*dict_t1[i]
    ortho_t2 += - np.einsum('ijab,ijab',dict_t2[i],t2_2)*dict_t2[i]
    ortho_So += - np.einsum('ijav,ijav',dict_So[i],So_2)*dict_So[i]
    ortho_Sv += - np.einsum('iuab,iuab',dict_Sv[i],Sv_2)*dict_Sv[i]
   
  for i in range(0,r+1):
    p = np.einsum('ia,ia',dict_t1[i],ortho_t1)
    q = np.einsum('ijab,ijab',dict_t2[i],ortho_t2)
    s = np.einsum('ijav,ijav',dict_So[i],ortho_So)
    t = np.einsum('iuab,iuab',dict_Sv[i],ortho_Sv)
    y= p+q+s+t
    #print p,q,s,t
    #print y
  
##--------Normalization of the new t and s------##
  '''  
  norm_t1 = t1_2/np.linalg.norm(t1_2)
  norm_t2 = t2_2/np.linalg.norm(t2_2)
  norm_So = So_2/np.linalg.norm(So_2)
  norm_Sv = Sv_2/np.linalg.norm(Sv_2)
  '''
  norm_t1 = ortho_t1/np.linalg.norm(ortho_t1)
  norm_t2 = ortho_t2/np.linalg.norm(ortho_t2)
  norm_So = ortho_So/np.linalg.norm(ortho_So)
  norm_Sv = ortho_Sv/np.linalg.norm(ortho_Sv)
  

##-------updating value of X for the next iteration-------##
  '''
  dict_t1[r+1] = ortho_t1
  dict_t2[r+1] = ortho_t2
  dict_So[r+1] = ortho_So
  dict_Sv[r+1] = ortho_Sv
  '''
  dict_t1[r+1] = norm_t1
  dict_t2[r+1] = norm_t2
  dict_So[r+1] = norm_So
  dict_Sv[r+1] = norm_Sv
     
  
   
