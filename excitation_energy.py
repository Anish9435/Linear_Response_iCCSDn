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
So = main.So
Sv = main.Sv
occ = MP2.occ
virt = MP2.virt
nao = MP2.nao
o_act = inp.o_act
v_act = inp.v_act
n_iter = inp.n_iter
n_davidson = inp.n_davidson
conv = 10**(-inp.conv)
D1 = -1.0*MP2.D1
D2 = -1.0*MP2.D2
Do = -1.0*MP2.Do
Dv = -1.0*MP2.Dv
t1_new,t2_new,So_new,Sv_new = davidson.guess_X(occ,virt,o_act,v_act)
twoelecint_mo = MP2.twoelecint_mo 

##-----------Initialization of Dictionaries for storing the values----------##

dict_Y_ia = {}
dict_Y_ijab = {}
dict_Y_ijav = {}
dict_Y_iuab = {}

dict_t1 = {}
dict_t2 = {}
dict_So = {}
dict_Sv = {}

##-----Storing the values in the dictionary--------##

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

  print ("*********")
  print ("iteration number "+str(x))
  r = x%n_davidson
  print ("Subspace vector "+str(r))
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
      
      dict_t1[r] = x_t1  
      dict_t2[r] = x_t2
      dict_So[r] = x_So
      dict_Sv[r] = x_Sv
    #print "dict_So" 
    #print dict_So[r] 
##------Diagrams and intermediates of coupled cluster theory i.e AX with new t and s---------##
  I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = intermediates.initialize()
##-------------Linear terms of both R_ia and R_ijab--------------##
  Y_ia = amplitude_response.singles_response_linear(I_oo,I_vv,dict_t1[r],dict_t2[r])
  Y_ijab = amplitude_response.doubles_response_linear(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,dict_t1[r],dict_t2[r])
##---------------------------------------------------------------##
##-------------update of the intermediates and geneation of new intermediates----------##
  I1, I2 = intermediates.R_ia_intermediates(t1)
  I1_new, I2_new = intermediates.R_ia_intermediates(dict_t1[r])
  I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = intermediates.update_int_response(t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)
  I_oo_new,I_vv_new,Ioooo_new,Iovvo_new,Iovvo_2_new,Iovov_new = intermediates.update_int_response(dict_t2[r],I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)
##-----------------------------------------------------------------##
##--------------one body diagrams---------------------------------##
  Y_ia += amplitude_response.singles_response_quadratic(I_oo,I_vv,I1,I2,dict_t1[r],dict_t2[r])
  Y_ia += amplitude_response.singles_response_quadratic(I_oo_new,I_vv_new,I1_new,I2_new,t1,t2)

  Y_ia += -2*np.einsum('ibkj,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,dict_t1[r])       #diagram non-linear a
  Y_ia +=  2*np.einsum('cbaj,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,dict_t1[r]) #diagram non-linear b
  Y_ia +=  np.einsum('ibjk,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,dict_t1[r])         #diagram non-linear c
  Y_ia += -np.einsum('cbja,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1,dict_t1[r])   #diagram non-linear d
  
  Y_ia += -2*np.einsum('ibkj,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],dict_t1[r],t1)
  Y_ia +=  2*np.einsum('cbaj,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],dict_t1[r],t1)
  Y_ia +=  np.einsum('ibjk,ka,jb->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],dict_t1[r],t1)
  Y_ia += -np.einsum('cbja,ic,jb->ia',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],dict_t1[r],t1)
##--------------------------------------------------------------------##
  I_oo,I_vv,I_oovo,I_vovv,Ioooo_2,I_voov,Iovov_3,Iovvo_3,Iooov,I3 = intermediates.singles_intermediates(t1,t2,I_oo,I_vv,I2)
  I_oo_new,I_vv_new,I_oovo_new,I_vovv_new,Ioooo_2_new,I_voov_new,Iovov_3_new,Iovvo_3_new,Iooov_new,I3_new = intermediates.singles_intermediates(dict_t1[r],dict_t2[r],I_oo_new,I_vv_new,I2_new)
##----------CC triple excitation diagram intermediate---------##
  II_oo = intermediates.W1_int_So(So)
  II_oo_new = intermediates.W1_int_So(dict_So[r])
  II_vv = intermediates.W1_int_Sv(Sv)
  II_vv_new = intermediates.W1_int_Sv(dict_Sv[r])
  #print "II_oo_new"
  #print II_oo_new
  #print "II_oo"
  #print II_oo
##------------------------------------------------------------##
##-----------------Two body diagrams--------------------------##
  Y_ijab += amplitude_response.doubles_response_quadratic(I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov,dict_t2[r]) 
  Y_ijab += amplitude_response.doubles_response_quadratic(I_oo_new,I_vv_new,Ioooo_new,Iovvo_new,Iovvo_2_new,Iovov_new,t2) 
  Y_ijab += amplitude_response.singles_n_doubles_response(dict_t1[r],I_oovo,I_vovv)
  Y_ijab += amplitude_response.singles_n_doubles_response(t1,I_oovo_new,I_vovv_new)

  Y_ijab += 0.5*np.einsum('ijkl,ka,lb->ijab',twoelecint_mo[:occ,:occ,:occ,:occ],t1,dict_t1[r])      #diagram non-linear 1
  Y_ijab += 0.5*np.einsum('cdab,ic,jd->ijab',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],t1,dict_t1[r])  #diagram non-linear 2
  Y_ijab += -np.einsum('ickb,ka,jc->ijab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,dict_t1[r])   #diagrams non-linear 3
  Y_ijab += -np.einsum('icak,jc,kb->ijab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,dict_t1[r])   #diagrams non-linear 4
  
  Y_ijab += 0.5*np.einsum('ijkl,ka,lb->ijab',twoelecint_mo[:occ,:occ,:occ,:occ],dict_t1[r],t1)      #diagram non-linear 1
  Y_ijab += 0.5*np.einsum('cdab,ic,jd->ijab',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],dict_t1[r],t1)  #diagram non-linear 2
  Y_ijab += -np.einsum('ickb,ka,jc->ijab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],dict_t1[r],t1)   #diagrams non-linear 3
  Y_ijab += -np.einsum('icak,jc,kb->ijab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],dict_t1[r],t1)   #diagrams non-linear 4

  print "Y_ijab"
  print Y_ijab
##---------------------A_lambda So and Sv sector------------##
  Y_ijab += amplitude_response.inserted_diag_So(dict_t2[r],II_oo)
  Y_ijab += amplitude_response.inserted_diag_So(t2,II_oo_new)
  Y_ijab += amplitude_response.inserted_diag_Sv(dict_t2[r],II_vv)
  Y_ijab += amplitude_response.inserted_diag_Sv(t2,II_vv_new)
  Y_ijab = cc_symmetrize.symmetrize(Y_ijab)

  print "Y_ijab"
  print Y_ijab
##------------------------------------------------------------##
##---------------------A_kappa Sv sector------------## 
  Y_iuab = amplitude_response.Sv_diagram_vs_contraction_response(dict_Sv[r])
##--------------------A_kappa T sector------------##
  #Y_iuab += amplitude_response.Sv_diagram_vt_contraction_response(dict_t2[r])
  #Y_iuab += amplitude_response.T1_contribution_Sv_response(dict_t1[r])
##---------------------A_kappa So sector------------## 
  Y_ijav = amplitude_response.So_diagram_vs_contraction_response(dict_So[r])
##----------------------A_kappa T sector------------##
  #Y_ijav += amplitude_response.So_diagram_vt_contraction_response(dict_t2[r])
  #Y_ijav += amplitude_response.T1_contribution_So_response(dict_t1[r])
  #print "Y_ijav"
  #print Y_ijav
##-------------------------------------------------------------------------------------------##
##-------Storing AX in the dictionary---------##
  
  dict_Y_ia[r] = Y_ia
  dict_Y_ijab[r] = Y_ijab
  dict_Y_ijav[r] = Y_ijav
  dict_Y_iuab[r] = Y_iuab

##---------Construction of B matrix---------------##

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
    B_Y_ia[m-1,r] = 2.0*np.einsum('ia,ia',dict_t1[m-1],dict_Y_ia[r])
    B_Y_ia[r,m-1] = 2.0*np.einsum('ia,ia',dict_t1[r],dict_Y_ia[m-1])

    B_Y_ijab[m-1,r] = 2.0*np.einsum('ijab,ijab',dict_t2[m-1],dict_Y_ijab[r])-np.einsum('ijba,ijab',dict_t2[m-1],dict_Y_ijab[r])
    B_Y_ijab[r,m-1] = 2.0*np.einsum('ijab,ijab',dict_t2[r],dict_Y_ijab[m-1])-np.einsum('ijba,ijab',dict_t2[r],dict_Y_ijab[m-1])

    B_Y_ijav[m-1,r] = 2.0*np.einsum('ijav,ijav',dict_So[m-1],dict_Y_ijav[r])-np.einsum('jiav,ijav',dict_So[m-1],dict_Y_ijav[r])
    B_Y_ijav[r,m-1] = 2.0*np.einsum('ijav,ijav',dict_So[r],dict_Y_ijav[m-1])-np.einsum('jiav,ijav',dict_So[r],dict_Y_ijav[m-1])

    B_Y_iuab[m-1,r] = 2.0*np.einsum('iuab,iuab',dict_Sv[m-1],dict_Y_iuab[r])-np.einsum('iuba,iuab',dict_Sv[m-1],dict_Y_iuab[r])
    B_Y_iuab[r,m-1] = 2.0*np.einsum('iuab,iuab',dict_Sv[r],dict_Y_iuab[m-1])-np.einsum('iuba,iuab',dict_Sv[r],dict_Y_iuab[m-1])

  B_Y_ia[r,r] = 2.0*np.einsum('ia,ia',dict_t1[r],dict_Y_ia[r])
  B_Y_ijab[r,r] = 2.0*np.einsum('ijab,ijab',dict_t2[r],dict_Y_ijab[r])-np.einsum('ijba,ijab',dict_t2[r],dict_Y_ijab[r])
  B_Y_ijav[r,r] = 2.0*np.einsum('ijav,ijav',dict_So[r],dict_Y_ijav[r])-np.einsum('jiav,ijav',dict_So[r],dict_Y_ijav[r])
  B_Y_iuab[r,r] = 2.0*np.einsum('iuab,iuab',dict_Sv[r],dict_Y_iuab[r])-np.einsum('iuba,iuab',dict_Sv[r],dict_Y_iuab[r])
   
  B_total = B_Y_ia+B_Y_ijab+B_Y_ijav+B_Y_iuab
  #print B_total

##-------Diagonalization of the B matrix----------##
      
  w_total, vects_total = np.linalg.eig(B_total)
  #print w_total
  
  if (np.all(w_total).imag <= 1e-8):
    w_total = w_total.real    
  #print w_total
  #print vects_total

##--Assigning the position of the lowest eigenvalue----##

  ind_min_wtotal = np.argmin(w_total)

##------Coefficient matrix corresponding to lowest eigenvalue-----##
    
  coeff_total = vects_total[:,ind_min_wtotal]
  #print coeff_total

  if (np.all(coeff_total).imag <= 1e-8):
    coeff_total = np.real(coeff_total)
    #print coeff_total
    #print 'after', coeff_total[i]
  
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

  lin_norm = 2.0*np.einsum('ia,ia',x_t1,x_t1)
  lin_norm += 2.0*np.einsum('ijab,ijab',x_t2,x_t2)-np.einsum('ijab,ijba',x_t2,x_t2)
  lin_norm += 2.0*np.einsum('ijav,ijav',x_So,x_So)-np.einsum('ijav,jiav',x_So,x_So)
  lin_norm += 2.0*np.einsum('iuab,iuab',x_Sv,x_Sv)-np.einsum('iuab,iuba',x_Sv,x_Sv)
  norm = math.sqrt(lin_norm)
 
  if (norm > 1e-9):
    x_t1 = x_t1/norm
    x_t2 = x_t2/norm
    x_So = x_So/norm
    x_Sv = x_Sv/norm

##------Formation of residual matrix--------##
  
  R_ia = np.zeros((occ,virt))
  R_ijab = np.zeros((occ,occ,virt,virt))      
  R_ijav = np.zeros((occ,occ,virt,o_act))
  R_iuab = np.zeros((occ,v_act,virt,virt))

  for i in range(0,r+1):
    R_ia += (coeff_total[i]*dict_Y_ia[i] - w_total[ind_min_wtotal]*coeff_total[i]*dict_t1[i])
    R_ijab += (coeff_total[i]*dict_Y_ijab[i] - w_total[ind_min_wtotal]*coeff_total[i]*dict_t2[i])
    R_ijav += (coeff_total[i]*dict_Y_ijav[i] - w_total[ind_min_wtotal]*coeff_total[i]*dict_So[i])
    R_iuab += (coeff_total[i]*dict_Y_iuab[i] - w_total[ind_min_wtotal]*coeff_total[i]*dict_Sv[i])

  eps_t = cc_update.update_t1t2(R_ia,R_ijab,x_t1,x_t2)[0]
  eps_So = cc_update.update_So(R_ijav,x_So)[0]
  eps_Sv = cc_update.update_Sv(R_iuab,x_Sv)[0]
  
  print 'EPS: ', eps_t 
  print 'Eigen value: ', w_total[ind_min_wtotal]
  if (eps_t <= conv and eps_So <= conv and eps_Sv <= conv):
    print "CONVERGED!!!!!!!!!!!!"
    print 'Excitation Energy: ', w_total[ind_min_wtotal]
    break

##--------Divide residue by denominator to get X--------##

  t1_2 = davidson.get_XO(R_ia,D1,w_total[ind_min_wtotal])
  t2_2 = davidson.get_XO(R_ijab,D2,w_total[ind_min_wtotal])
  So_2 = davidson.get_XO(R_ijav,Do,w_total[ind_min_wtotal])
  Sv_2 = davidson.get_XO(R_iuab,Dv,w_total[ind_min_wtotal])

##------Schmidt orthogonalization----------##
  
  ortho_t1 = t1_2
  ortho_t2 = t2_2
  ortho_So = So_2
  ortho_Sv = Sv_2
  
  for i in range(0,r+1):
    ovrlap = 2.0*np.einsum('ia,ia',t1_2,dict_t1[i]) + 2.0*np.einsum('ijab,ijab',t2_2,dict_t2[i]) - np.einsum('ijab,ijba',t2_2,dict_t2[i]) + 2.0*np.einsum('ijav,ijav',So_2,dict_So[r]) - np.einsum('ijav,jiav',So_2,dict_So[r]) + 2.0*np.einsum('iuab,iuab',Sv_2,dict_Sv[r]) - np.einsum('iuab,iuba',Sv_2,dict_Sv[r])

    ortho_t1 += -ovrlap*dict_t1[i]  
    ortho_t2 += -ovrlap*dict_t2[i]  
    ortho_So += -ovrlap*dict_So[i]  
    ortho_Sv += -ovrlap*dict_Sv[i]  
  #for i in range(0,r+1):
  #  p = 2.0*np.einsum('ia,ia',dict_t1[i],ortho_t1)
  #  q = 2.0*np.einsum('ijab,ijab',dict_t2[i],ortho_t2)-np.einsum('ijab,ijba',dict_t2[i],ortho_t2)
  #  t = 2.0*np.einsum('ijav,ijav',dict_So[i],ortho_So)-np.einsum('ijav,jiav',dict_So[i],ortho_So)
  #  d = 2.0*np.einsum('iuab,iuab',dict_Sv[i],ortho_Sv)-np.einsum('iuab,iuba',dict_Sv[i],ortho_Sv)
  #  y = p+q+t+d
  #  print "overlap:", i,p,q,t,d,y  
 
##--------Normalization of the new t and s------##

  ortho_norm = 2.0*np.einsum('ia,ia',ortho_t1,ortho_t1)
  ortho_norm += 2.0*np.einsum('ijab,ijab',ortho_t2,ortho_t2)-np.einsum('ijab,ijba',ortho_t2,ortho_t2)
  ortho_norm += 2.0*np.einsum('ijav,ijav',ortho_So,ortho_So)-np.einsum('ijav,jiav',ortho_So,ortho_So)
  ortho_norm += 2.0*np.einsum('iuab,iuab',ortho_Sv,ortho_Sv)-np.einsum('iuab,iuba',ortho_Sv,ortho_Sv)
  norm_total = math.sqrt(ortho_norm)

  if (norm_total > 1e-9):
    norm_t1 = ortho_t1/norm_total
    norm_t2 = ortho_t2/norm_total
    norm_So = ortho_So/norm_total
    norm_Sv = ortho_Sv/norm_total
  else:  
    print 'Error in calculation: Generating vector with zero norm'
    quit()
  
##-------updating value of X for the next iteration-------##
 
  dict_t1[r+1] = norm_t1
  dict_t2[r+1] = norm_t2
  dict_So[r+1] = norm_So
  dict_Sv[r+1] = norm_Sv

  nrm = 2.0*np.einsum('ia,ia',dict_t1[r+1],dict_t1[r+1]) + 2.0*np.einsum('ijab,ijab',dict_t2[r+1],dict_t2[r+1]) - np.einsum('ijab,ijba',dict_t2[r+1],dict_t2[r+1]) + 2.0*np.einsum('ijav,ijav',dict_So[r+1],dict_So[r+1]) - np.einsum('ijav,jiav',dict_So[r+1],dict_So[r+1]) + 2.0*np.einsum('iuab,iuab',dict_Sv[r+1],dict_Sv[r+1]) - np.einsum('iuab,iuba',dict_Sv[r+1],dict_Sv[r+1])
  print "final norm:", nrm 
     
