from pyscf import scf
import numpy as np
import copy as cp
import davidson
import inp_he as inp
import MP2
import intermediates
import amplitude_cisd
import cc_symmetrize
import trans_mo
import cc_update
import main
import math

##commenting here for git 
##------Import important values-------##

#m = scf.RHF(inp.mol)
#ehf = m.scf() 
ehf = main.E_hf
print 'Hartree-Fock energy: ', ehf
c1 = main.t1
c2 = main.t2

occ = MP2.occ
virt = MP2.virt

t0 = 1.0
t1 = np.zeros((occ,virt))
t2 = np.zeros((occ,occ,virt,virt))      

Fock_mo = MP2.Fock_mo
twoelecint_mo = MP2.twoelecint_mo 
oneelecint_mo = trans_mo.oneelecint_mo
#print 'OneElecint:'
#print oneelecint_mo
nao = MP2.nao
#Fock_mo = oneelecint_mo

o_act = inp.o_act
v_act = inp.v_act
n_iter = inp.n_iter
n_davidson = inp.n_davidson
conv = 10**(-inp.conv)

Enuc = 'Nuclear energy', inp.mol.energy_nuc()

def write_t1(tmat,occ):
  len_1 = len(tmat[:,0])
  len_2 = len(tmat[0,:])

  for i in range(len_1):
    for j in range(len_2):
      if (abs(tmat[i,j])>1e-12):
        #print '%4d%4d%18.10E\n' % (i+1,j+1+occ,tmat[i,j])
        break

def write_t2(tmat,occ):
  len_1 = len(tmat[:,0,0,0])
  len_2 = len(tmat[0,:,0,0])
  len_3 = len(tmat[0,0,:,0])
  len_4 = len(tmat[0,0,0,:])

  for i in range(len_1):
    for j in range(len_2):
      for k in range(len_3):
        for l in range(len_4):
          if (abs(tmat[i,j,k,l])>1e-12):
            #print '%4d%4d%4d%4d%18.10E\n' % (i+1,j+1,k+1+occ,l+1+occ,tmat[i,j,k,l])
            break

def get_preconditioner(occ,virt,nao, Fock_mo, twoelecint_mo, ehf):
  
  D1 = np.zeros((occ,virt))
  D2 = np.zeros((occ,occ,virt,virt))
  
  for i in range(0,occ):
    for j in range(0,occ):
      for a in range(occ,nao):
        for b in range(occ,nao):
          D2[i,j,a-occ,b-occ] = ehf - Fock_mo[i,i] - Fock_mo[j,j] + Fock_mo[a,a] + Fock_mo[b,b]
          D2[i,j,a-occ,b-occ] += 0.5*(twoelecint_mo[a,a,a,a]+twoelecint_mo[b,b,b,b]-twoelecint_mo[i,i,i,i]-twoelecint_mo[j,j,j,j])
          for k in range(0,occ):
            if (i!=k):
              D2[i,j,a-occ,b-occ] += 2*(twoelecint_mo[a,k,a,k] - twoelecint_mo[i,k,i,k]) - twoelecint_mo[a,k,k,a] + twoelecint_mo[i,k,k,i]
            if (j!=k):
              D2[i,j,a-occ,b-occ] += 2*(twoelecint_mo[b,k,b,k] - twoelecint_mo[j,k,j,k]) - twoelecint_mo[b,k,k,b] + twoelecint_mo[j,k,k,j]

  for i in range(0,occ):
    for a in range(occ,nao):
          D1[i,a-occ] = ehf - Fock_mo[i,i] + Fock_mo[a,a]
          D1[i,a-occ] += twoelecint_mo[i,a,i,a]-twoelecint_mo[i,i,i,i]
          for k in range(0,occ):
            if (i!=k):
              D1[i,a-occ] += 2*(twoelecint_mo[a,k,a,k] - twoelecint_mo[i,k,i,k]) - twoelecint_mo[a,k,k,a] + twoelecint_mo[i,k,k,i]

  return D1, D2

D1, D2 = get_preconditioner(occ,virt,nao, oneelecint_mo, twoelecint_mo, ehf)
#print '1-electron Preconditioner'
write_t1(D1, occ)
#print '2-electron Preconditioner'
write_t2(D2, occ)

#R_ia = cp.deepcopy(Fock_mo[:occ,occ:nao])
#R_ijab = 0.5*cp.deepcopy(twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
#
#t1 = davidson.get_X(R_ia,D1)
#t2 = davidson.get_X(R_ijab,D2)

sq_norm = 0
sq_norm += 1.0
sq_norm += np.einsum('ia,ia',t1,t1)
sq_norm += np.einsum('ijab,ijab',t2,t2)
norm = math.sqrt(sq_norm)

##-----------Initialization of Dictionaries for storing the values----------##

dict_Y_0 = {}
dict_Y_ia = {}
dict_Y_ijab = {}

dict_t0 = {}
dict_t1 = {}
dict_t2 = {}

#Store the norm of each of the Ritz vectors
arr_norm_t=[]

##-----Storing the values in the dictionary--------##

dict_t0[0] = t0
dict_t1[0] = t1
dict_t2[0] = t2
arr_norm_t.append(1.0)

##-----Initialization of the 1*1 B matrix-------##

B_Y_0 = np.zeros((1,1))
B_Y_ia = np.zeros((1,1))
B_Y_ijab = np.zeros((1,1))

##------Start iteration--------##
#for x in range(0,n_iter):
for x in range(0,4):

##-----conditioning with the remainder------##

  print ("*********")
  print ("iteration number "+str(x))
  r = x%n_davidson
  print ("Subspace vector "+str(r))
  if(x>0):
    if r==0:
      dict_Y_0.clear()
      dict_Y_ia.clear()
      dict_Y_ijab.clear()

      dict_t0.clear()
      dict_t1.clear()
      dict_t2.clear()
      
      B_Y_0 = np.zeros((r+1,r+1))
      B_Y_ia = np.zeros((r+1,r+1))
      B_Y_ijab = np.zeros((r+1,r+1))

      dict_t0[r] = x_t0 # This should be the proper way, anyway convergence is still hard to achieve even with this. Need to look in to it. 
      dict_t1[r] = x_t1 # This should be the proper way, anyway convergence is still hard to achieve even with this. Need to look in to it. 
      dict_t2[r] = x_t2
  
##------Diagrams of coupled cluster theory i.e AX with new t and s---------##
  
  tau = cp.deepcopy(dict_t2[r]) 
  
  I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2, I_oovo, I_vovv = intermediates.initialize()
  I1, I2 = intermediates.R_ia_intermediates(t1)
  
  Y_0 = amplitude_cisd.zero(ehf,dict_t0[r],dict_t1[r],dict_t2[r])
  Y_ia = amplitude_cisd.singles(I1,I2,I_oo,I_vv,tau,dict_t0[r],dict_t1[r],dict_t2[r],ehf)

  #print 'Printing Y_ia'
  #print write_t1(Y_ia,occ)

  Y_ijab = amplitude_cisd.doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,I_oovo,I_vovv,dict_t0[r],dict_t1[r],tau,dict_t2[r],ehf)
  #Y_ijab += amplitude.singles_n_doubles_cisd(dict_t1[r],dict_t2[r],tau,I_oovo,I_vovv)
 
  Y_ijab = cc_symmetrize.symmetrize(Y_ijab)
 
  #print 'Printing Y_ijab'
  #print write_t2(Y_ijab,occ)
  

##-------Storing AX in the dictionary---------##
      
  dict_Y_0[r] = Y_0
  dict_Y_ia[r] = Y_ia
  dict_Y_ijab[r] = Y_ijab

 #print 'Y_ijab'
 #print Y_ijab

##---------Construction of B matrix---------------##

  #print "----B Matrix-------" 
  B_Y_0_nth = np.zeros((r+1,r+1))
  B_Y_ia_nth = np.zeros((r+1,r+1))
  B_Y_ijab_nth = np.zeros((r+1,r+1))
  
  B_Y_0_nth[:r,:r] = B_Y_0
  B_Y_ia_nth[:r,:r] = B_Y_ia 
  B_Y_ijab_nth[:r,:r] = B_Y_ijab
  
  B_Y_0 = cp.deepcopy(B_Y_0_nth)
  B_Y_ia = cp.deepcopy(B_Y_ia_nth)
  B_Y_ijab = cp.deepcopy(B_Y_ijab_nth)
  
  B_Y_0_nth = None
  B_Y_ia_nth = None
  B_Y_ijab_nth = None
 

  for m in range(1,r+1):
    #print '*****'
    B_Y_0[m-1,r] = dict_t0[m-1]*dict_Y_0[r]
    #print dict_t0[m-1],dict_Y_0[r],B_Y_0[m-1,r]
    B_Y_0[r,m-1] = dict_t0[r]*dict_Y_0[m-1]
    #print B_Y_0[r,m-1]
    #print '*****'

    B_Y_ia[m-1,r] = np.einsum('ia,ia',dict_t1[m-1],dict_Y_ia[r])
    B_Y_ia[r,m-1] = np.einsum('ia,ia',dict_t1[r],dict_Y_ia[m-1])

    #print '*****'
    B_Y_ijab[m-1,r] = np.einsum('ijab,ijab',dict_t2[m-1],dict_Y_ijab[r])
    #print B_Y_ijab[m-1,r]
    #print 'Printing dict_t2[',m-1,']'
    #write_t2(dict_t2[m-1], occ)
    #print 'Pr_Y_ijab[',r']'
    #write_t2(dict_Y_ijab[r], occ)
    B_Y_ijab[r,m-1] = np.einsum('ijab,ijab',dict_t2[r],dict_Y_ijab[m-1])
    #print B_Y_ijab[r,m-1]
    #print 'Printing dict_t2[',r,']'
    #write_t2(dict_t2[r], occ)
    #print 'Pr_Y_ijab[',m-1,']'
    #write_t2(dict_Y_ijab[m-1], occ)
    #print '*****'

  B_Y_0[r,r] = dict_t0[r]*dict_Y_0[r]
  B_Y_ia[r,r] = np.einsum('ia,ia',dict_t1[r],dict_Y_ia[r])
  B_Y_ijab[r,r] = np.einsum('ijab,ijab',dict_t2[r],dict_Y_ijab[r])
   
  #B_total = B_Y_ia+B_Y_ijab+B_Y_iuab+B_Y_ijav
  B_total = B_Y_0+B_Y_ia+B_Y_ijab
  #print 'Print B Matrices:'
  print B_Y_0
  print '\n'
  print B_Y_ia
  print '\n'
  print B_Y_ijab
  print '\n'
  print B_total
  print '\n'

##-------Diagonalization of the B matrix----------##
      
  w_total, vects_total = np.linalg.eig(B_total)
  print w_total
  #print vects_total
 
##--Assigning the position of the lowest eigenvalue----##

  ind_min_wtotal = np.argmin(w_total)
##------Coefficient matrix corresponding to lowest eigenvalue-----##
    
  coeff_total = vects_total[:,ind_min_wtotal]
  print coeff_total

##------Linear Combination--------------##

  x_t0 = 0.0
  x_t1 = np.zeros((occ,virt))
  x_t2 = np.zeros((occ,occ,virt,virt))      ###x_t1 and so on are the linear combination of the coefficient vector elements and the t1,t1,So and Sv values of successive iterations###

  # x_t1 and x_t2 will only be required while starting a new subspace, so it should be done only for those special iterations
  for i in range(0,r+1):
    x_t0 += coeff_total[i]*dict_t0[i]
    x_t1 += np.linalg.multi_dot([coeff_total[i],dict_t1[i]])
    x_t2 += np.linalg.multi_dot([coeff_total[i],dict_t2[i]])

  sq_norm = 0
  sq_norm += x_t0*x_t0
  sq_norm += np.einsum('ia,ia',x_t1,x_t1) 
  sq_norm += np.einsum('ijab,ijab',x_t2,x_t2) 
  norm = math.sqrt(sq_norm)
  
  if (norm > 1e-9):
    x_t0 = x_t0/norm
    x_t1 = x_t1/norm
    x_t2 = x_t2/norm

##------Formation of residual matrix--------##
  
  R_0 = 0.0
  R_ia = np.zeros((occ,virt))
  R_ijab = np.zeros((occ,occ,virt,virt))      

  for i in range(0,r+1): 
    R_0 += (coeff_total[i]*dict_Y_0[i] - w_total[ind_min_wtotal]*coeff_total[i]*dict_t0[i])
    R_ia += (coeff_total[i]*dict_Y_ia[i] - w_total[ind_min_wtotal]*coeff_total[i]*dict_t1[i])
    R_ijab += (coeff_total[i]*dict_Y_ijab[i] - w_total[ind_min_wtotal]*coeff_total[i]*dict_t2[i])

  print 'Residue'
  write_t1(R_ia,occ)
  write_t2(R_ijab,occ)

  eps_t = abs(R_0) + cc_update.update_t1t2(R_ia,R_ijab,x_t1,x_t2)[0]
  #eps_t = cc_update.update_t1t2(R_ia,R_ijab,x_t1,x_t2)[0]
  
  print 'EPS: ', eps_t 
  print 'Eigen value: ', w_total[ind_min_wtotal]
  omega = w_total[ind_min_wtotal]
  if (eps_t <= conv):
    print "CONVERGED!!!!!!!!!!!!"
    print 'Excitation Energy: ', w_total[ind_min_wtotal]
    break

##--------Divide residue by denominator to get X--------##
## Using a different denominator that includes Omega 

  t0_2 = R_0
  t1_2 = davidson.get_XO(R_ia,D1,omega)
  t2_2 = davidson.get_XO(R_ijab,D2,omega)

  print 'Before Orthogonalization'
  write_t2(t2_2,occ)
  
  sq_norm = 0.0
  sq_norm += t0_2*t0_2
  sq_norm += np.einsum('ia,ia',t1_2,t1_2)
  sq_norm += np.einsum('ijab,ijab',t2_2,t2_2)
  norm = math.sqrt(sq_norm)
  print 'xnrm: ', norm

  t0 = t0_2/norm
  t1 = t1_2/norm
  t2 = t2_2/norm

##------Schmidt orthogonalization----------##

 #ortho_t0 = R_0
 #ortho_t1 = davidson.get_X(R_ia,D1)
 #ortho_t2 = davidson.get_X(R_ijab,D2)
 #ortho_t1 = davidson.get_XO(R_ia,D1,omega)
 #ortho_t2 = davidson.get_XO(R_ijab,D2,omega)

  ortho_t0 = t0_2/norm
  ortho_t1 = t1_2/norm
  ortho_t2 = t2_2/norm

  print 'Printing norm'
  print arr_norm_t
  for i in range(0,r+1):
    ortho_t0 += dict_t0[i]*t0*dict_t0[i]#/arr_norm_t[i]
    ortho_t1 += - (np.einsum('ia,ia',dict_t1[i],t1)*dict_t1[i])#/arr_norm_t[i]
    ortho_t2 += - (np.einsum('ijab,ijab',dict_t2[i],t2)*dict_t2[i])#/arr_norm_t[i]
   
  print 'After Orthogonalization'
  write_t2(ortho_t2,occ)

##--------Normalization of the new t and s------##
  '''  
  norm_t1 = t1_2/np.linalg.norm(t1_2)
  norm_t2 = t2_2/np.linalg.norm(t2_2)
  '''
 #norm_total = np.linalg.norm(ortho_t1) + np.linalg.norm(ortho_t2)
 #print 'NORM_TOTAL:', norm_total

  # The new Ritz vector is normalized here. First the total norm is calculated for the orthogonalized vector

  sq_norm = 0.0
  sq_norm += ortho_t0*ortho_t0
  sq_norm += np.einsum('ia,ia',ortho_t1,ortho_t1)
  sq_norm += np.einsum('ijab,ijab',ortho_t2,ortho_t2)
  norm = math.sqrt(sq_norm)

  # Each part of the vector is divided by the total norm of the vector
  if (norm > 1e-9):
    norm_t0 = ortho_t0/norm
    norm_t1 = ortho_t1/norm
    norm_t2 = ortho_t2/norm
  else:  
    print 'Error in calculation: Generating vector with zero norm'
    quit()

##-------updating value of X for the next iteration-------##
  dict_t0[r+1] = norm_t0
  dict_t1[r+1] = norm_t1
  dict_t2[r+1] = norm_t2

  arr_norm_t.append(norm)
