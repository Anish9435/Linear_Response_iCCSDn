
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

def W1_int_So(So):
  II_oo = np.zeros((occ,occ)) 
  II_oo[:,occ-o_act:occ] += -2*0.25*np.einsum('ciml,mlcv->iv',twoelecint_mo[occ:nao,:occ,:occ,:occ],So) + 0.25*np.einsum('diml,lmdv->iv',twoelecint_mo[occ:nao,:occ,:occ,:occ],So) 
  return II_oo
  gc.collect()

def W1_int_Sv(Sv):
  II_vv = np.zeros((virt,virt))
  II_vv[:v_act,:] += 2*0.25*np.einsum('dema,mude->ua',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],Sv) - 0.25*np.einsum('dema,mued->ua',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],Sv)
  return II_vv
  gc.collect()

#     Create intermediates for contribution of singles to R_ijab  
def R_ia_intermediates(t1):
  I1 = 2*np.einsum('cbkj,kc->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1)
  I2 = -np.einsum('cbjk,kc->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1)
  return I1,I2
  I1 = None
  I2 = None

def singles_intermediates_response(t1,t2,I_oo,I_vv):
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

  return I_oo, I_vv, I_oovo, I_vovv
  I_vv = None
  I_oo = None
  I_oovo = None
  I_vovv = None
  gc.collect()

##----------------------------------------------------------------------------------------------##
                ##Intermediates for (V_S_T)_c terms contributing to R_iuab and R_ijav##
##----------------------------------------------------------------------------------------------##

def coupling_terms_Sv_response(Sv):
  II_vo = np.zeros((virt,o_act))
  II_vo[:v_act,:] += 2*0.25*np.einsum('cblv,lwcb->wv',twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],Sv) - 0.25*np.einsum('bclv,lwcb->wv',twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],Sv)
 
  return II_vo
  gc.collect()

def coupling_terms_So_response(So):
  II_ov = np.zeros((v_act,occ)) 
  II_ov[:,occ-o_act:occ] += -2*0.25*np.einsum('dulk,lkdx->ux',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],So) + 0.25*np.einsum('dulk,kldx->ux',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],So) 
  
  return II_ov
  gc.collect()

##-----------------------------------------------------------------------------------------------------------##
                ##Two body Intermediates for (V_S_T)_c terms contributing to R_iuab and R_ijav##
##-----------------------------------------------------------------------------------------------------------##

def w2_int_So_response(So):
  II_ovoo = np.zeros((occ,virt,o_act,occ))
  II_ovoo3 = np.zeros((occ,v_act,occ,occ))
  II_vvvo3 = np.zeros((virt,v_act,virt,occ))

  II_ovoo[:,:,:,occ-o_act:occ] += -np.einsum('cdvk,jkcw->jdvw',twoelecint_mo[occ:nao,occ:nao,occ-o_act:occ,:occ],So)

##-----------------------------------------------------------------------------------------------------------##
                     ##Intermediates for off diagonal terms like So->R_iuab##
##-----------------------------------------------------------------------------------------------------------##

  II_ovoo3[:,:,:,occ-o_act:occ] += -np.einsum('dulk,ikdw->iulw',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],So)
  II_vvvo3[:,:,:,occ-o_act:occ] += -np.einsum('dulk,lkaw->duaw',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],So)
 
  return II_ovoo,II_ovoo3,II_vvvo3
  gc.collect()


def w2_int_Sv_response(Sv):
  II_vvvo = np.zeros((v_act,virt,virt,occ))
  II_vvvo2 = np.zeros((virt,virt,virt,o_act))
  II_ovoo2 = np.zeros((occ,virt,occ,o_act))
 
  II_vvvo[:,:v_act,:,:] += -np.einsum('uckl,kxbc->uxbl',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,:occ],Sv) 

##-----------------------------------------------------------------------------------------------------------##
                     ##Intermediates for off diagonal terms like Sv->R_ijav##
##-----------------------------------------------------------------------------------------------------------##

  II_vvvo2[:,:v_act,:,:] += -np.einsum('dckv,kxac->dxav',twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],Sv)
  II_ovoo2[:,:v_act,:,:] += np.einsum('dckv,ixdc->ixkv',twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],Sv)

  return II_vvvo,II_vvvo2,II_ovoo2
  gc.collect()


                          ##-----------------------------------------------------------------------------------------------------------------------##
                                                                                    #THE END#
                          ##-----------------------------------------------------------------------------------------------------------------------##


