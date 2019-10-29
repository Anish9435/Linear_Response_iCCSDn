# Import modules
import gc
import numpy as np
import copy as cp
import trans_mo
import MP2
import inp

mol = inp.mol

# import important stuff
twoelecint_mo = MP2.twoelecint_mo 
Fock_mo = MP2.Fock_mo
occ = MP2.occ
virt = MP2.virt
nao = MP2.nao
 
#      active orbital numbers imported from input file
o_act = inp.o_act
v_act = inp.v_act
act = o_act + v_act

##--------Initialize intermediates for linear terms in CCD. LCCD is the default calculation------##

def initialize():
  I_vv = cp.deepcopy(Fock_mo[occ:nao,occ:nao])
  I_oo = cp.deepcopy(Fock_mo[:occ,:occ])
  Ivvvv = cp.deepcopy(twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao])
  Ioooo = cp.deepcopy(twoelecint_mo[:occ,:occ,:occ,:occ])
  Iovvo = cp.deepcopy(twoelecint_mo[:occ,occ:nao,occ:nao,:occ])
  Iovvo_2 = cp.deepcopy(twoelecint_mo[:occ,occ:nao,occ:nao,:occ])
  Iovov = cp.deepcopy(twoelecint_mo[:occ,occ:nao,:occ,occ:nao])
  Iovov_2 = cp.deepcopy(twoelecint_mo[:occ,occ:nao,:occ,occ:nao])
  return I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov, Iovov_2
  I_vv = None
  I_oo = None
  Ivvvv = None
  Ioooo = None
  Iovvo = None
  Iovvo_2 = None
  Iovov = None
  Iovov_2 = None
  I_oovo = None
  I_vovv = None
  gc.collect()

  
##--------Introducing the non-linear doubles terms-----------##

def update_int(tau,t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov):
  I_vv += -2*np.einsum('cdkl,klad->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) + np.einsum('cdkl,klda->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)

  I_oo += 2*np.einsum('cdkl,ilcd->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dckl,lidc->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) ##t2 should be changed to tau here

  Ioooo += np.einsum('cdkl,ijcd->ijkl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)

  Iovvo += np.einsum('dclk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - 0.5*np.einsum('cdlk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)

  Iovvo_2 += -0.5*np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)  - np.einsum('dckl,ljdb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)
 
  Iovov += -0.5*np.einsum('dckl,ildb->ickb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)
  
  return I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov
  I_vv = None
  I_oo = None
  Iovvo = None
  Iovvo_2 = None
  Iovov = None
  gc.collect()
  
##--------For Linear Response Theory-------##
  
def update_int_response(t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov):
  I_vv = -2*np.einsum('cdkl,klad->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) + np.einsum('cdkl,klda->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)

  I_oo = 2*np.einsum('cdkl,ilcd->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dckl,lidc->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)

  Ioooo = np.einsum('cdkl,ijcd->ijkl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)

  Iovvo = np.einsum('dclk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - 0.5*np.einsum('cdlk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)

  Iovvo_2 = -0.5*np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)  - np.einsum('dckl,ljdb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)
 
  Iovov = -0.5*np.einsum('dckl,ildb->ickb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)
  
  return I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov
  I_vv = None
  I_oo = None
  Ioooo = None
  Iovvo_2 = None
  Iovov = None
  gc.collect()

##-------Creating intermediates involving S which will lead to diagrams in R_ijab-------##

def W1_int_So(x):
  II_oo = np.zeros((occ,occ)) 
  II_oo[:,occ-o_act:occ] += -2*0.25*np.einsum('ciml,mlcv->iv',twoelecint_mo[occ:nao,:occ,:occ,:occ],x) + 0.25*np.einsum('diml,lmdv->iv',twoelecint_mo[occ:nao,:occ,:occ,:occ],x)
  return II_oo
  gc.collect()

def W1_int_Sv(x):
  II_vv = np.zeros((virt,virt))
  II_vv[:v_act,:] += 2*0.25*np.einsum('dema,mude->ua',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],x) - 0.25*np.einsum('dema,mued->ua',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],x)
  return II_vv
  gc.collect()

#     Create intermediates for contribution of singles to R_ijab  
def R_ia_intermediates(t1):
  I1 = 2*np.einsum('cbkj,kc->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1)
  I2 = -np.einsum('cbjk,kc->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1)
  return I1,I2
  I1 = None
  I2 = None
  
  
def singles_intermediates(t1,t2,I_oo,I_vv,I2):
  I_oo += 2*np.einsum('ibkj,jb->ik',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1)    #intermediate for diagrams 5
  I_oo += -np.einsum('ibjk,jb->ik',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1)     #intermediate for diagrams 8
  I_vv += 2*np.einsum('bcja,jb->ca',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1)    #intermediate for diagrams 6
  I_vv += -np.einsum('cbja,jb->ca',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1)    #intermediate for diagrams 7
  #I_vv += -2*np.einsum('dclk,ld,ka->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)  #intermediate for diagram 34'
  
  I_oovo = np.zeros((occ,occ,virt,occ))
  I_oovo += -np.einsum('cikl,jlca->ijak',twoelecint_mo[occ:nao,:occ,:occ,:occ],t2)    #intermediate for diagrams 11
  I_oovo += np.einsum('cdka,jicd->ijak',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2)    #intermediate for diagrams 12
  I_oovo += -np.einsum('jclk,lica->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)    #intermediate for diagrams 13
  I_oovo += 2*np.einsum('jckl,ilac->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)    #intermediate for diagrams 15
  I_oovo += -np.einsum('jckl,ilca->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)    #intermediate for diagrams 17
  
  I_vovv = np.zeros((virt,occ,virt,virt))
  I_vovv += np.einsum('cjkl,klab->cjab',twoelecint_mo[occ:nao,:occ,:occ,:occ],t2)    #intermediate for diagrams 9
  I_vovv += -np.einsum('cdlb,ljad->cjab',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2)    #intermediate for diagrams 10
  I_vovv += -np.einsum('cdka,kjdb->cjab',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2)    #intermediate for diagrams 14
  I_vovv += 2*np.einsum('cdal,ljdb->cjab',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2)    #intermediate for diagrams 16
  I_vovv += -np.einsum('cdal,jldb->cjab',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2)    #intermediate for diagrams 18

  Ioooo_2 = 0.5*np.einsum('cdkl,ic,jd->ijkl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)    #intermediate for diagrams 37
  I_voov = -np.einsum('cdkl,kjdb->cjlb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)    #intermediate for diagrams 39

  Iovov_3 = -np.einsum('dckl,ildb->ickb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)  #intermediate for diagrams 36
  
  Iovvo_3 = 2*np.einsum('dclk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) + np.einsum('cdak,ic->idak',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1)  #intermediate for diagrams 32,33,31
  
  Iooov = np.einsum('dl,ijdb->ijlb',I2,t2) #intermediate for diagram 34

  Iovvo_3 += -np.einsum('iclk,la->icak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1)  #intermediate for diagram 30
  
  I3 = -np.einsum('cdkl,ic,ka->idal',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)  #intermediate for diagram 40

  return I_oo, I_vv, I_oovo, I_vovv, Ioooo_2, I_voov, Iovov_3, Iovvo_3, Iooov, I3
  I_vv = None
  I_oo = None
  I_oovo = None
  I_vovv = None
  Ioooo_2 = None
  I_voov = None
  Iovov_3 = None
  Iovvo_3 = None
  Iooov = None
  I3 = None
  gc.collect()
  

