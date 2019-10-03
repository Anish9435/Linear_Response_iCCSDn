# Import modules
import gc
import numpy as np
import copy as cp
import trans_mo
import MP2
import inp
import intermediates

##-------import important stuff-------##

nao = MP2.nao
twoelecint_mo = MP2.twoelecint_mo 
Fock_mo = MP2.Fock_mo
D1 = MP2.D1
D2=MP2.D2
So = MP2.So
Do=MP2.Do
Sv = MP2.Sv
Dv=MP2.Dv
occ = MP2.occ
virt = MP2.virt
o_act = inp.o_act
v_act = inp.v_act
conv = 10**(-inp.conv)


##-------Function exclusive for LRT i.e there is no bare hamiltonian term------##

def singles_response_linear(I_oo,I_vv,t1,t2):
  R_ia = -np.einsum('ik,ka->ia',I_oo,t1)   #diagrams 1
  I_oo = None
  R_ia += np.einsum('ca,ic->ia',I_vv,t1)   #diagrams 2
  I_vv = None
  R_ia += 2*np.einsum('icak,kc->ia',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1)    #diagram 3
  R_ia += -np.einsum('icka,kc->ia',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1)     #diagram 4
  R_ia += -2*np.einsum('ibkj,kjab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)     #diagrams 5 
  R_ia += np.einsum('ibkj,jkab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)     #diagrams 6 
  R_ia += 2*np.einsum('cdak,ikcd->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2) #diagrams 7 
  R_ia += -np.einsum('cdak,ikdc->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2) #diagrams 8 
  return R_ia
  R_ia = None
  gc.collect()

##----------Quardratic terms--------------##

def singles_response_quadratic(I_oo,I_vv,I1,I2,t1,t2):
  R_ia = -np.einsum('ik,ka->ia',I_oo,t1)   #diagrams l,j,m,n
  I_oo = None
  R_ia += np.einsum('ca,ic->ia',I_vv,t1)   #diagrams k,i
  I_vv = None
  R_ia += 2*np.einsum('bj,ijab->ia',I1,t2) - np.einsum('bj,ijba->ia',I1,t2)     #diagrams e,f
  I1 = None
  R_ia += 2*np.einsum('bj,ijab->ia',I2,t2) - np.einsum('bj,ijba->ia',I2,t2)     #diagrams g,h
  I2 = None
  return R_ia
  R_ia = None
  gc.collect()

##-------Similar as R_ia for LRT i.e no bare hamiltonian term---------##

def doubles_response_linear(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,t1,t2):
  print " "
  R_ijab = -np.einsum('ik,kjab->ijab',I_oo,t2)  #diagram 1
  I_oo = None
  R_ijab += np.einsum('ca,ijcb->ijab',I_vv,t2)  #diagram 2
  I_vv = None
  R_ijab += -np.einsum('ijkb,ka->ijab',twoelecint_mo[:occ,:occ,:occ,occ:nao],t1)      #diagram 3
  R_ijab += np.einsum('cjab,ic->ijab',twoelecint_mo[occ:nao,:occ,occ:nao,occ:nao],t1) #diagram 4
  R_ijab += 0.5*np.einsum('cdab,ijcd->ijab',Ivvvv,t2)  #diagram 5
  Ivvvv = None
  R_ijab += 2*np.einsum('jcbk,kica->ijab',Iovvo,t2)      #diagram 6 
  Iovvo =None
  R_ijab += -np.einsum('icka,kjcb->ijab',Iovov_2,t2)  #diagram 7 
  Iovov_2 = None
  R_ijab += - np.einsum('jcbk,ikca->ijab',Iovvo_2,t2)  #diagram 8  
  Iovvo_2 = None
  R_ijab += 0.5*np.einsum('ijkl,klab->ijab',Ioooo,t2)  #diagram 9 
  Ioooo = None
  R_ijab += - np.einsum('ickb,kjac->ijab',Iovov,t2)    #linear 10 
  Iovov = None
  return R_ijab
  R_ijab = None
  gc.collect()

##--------Quadratic terms-------------##

def doubles_response_quadratic(I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov,t2):
  print " "
  R_ijab = -np.einsum('ik,kjab->ijab',I_oo,t2)     #diagrams 25,27,5,8
  I_oo = None
  R_ijab += np.einsum('ca,ijcb->ijab',I_vv,t2)     #diagrams 24,26,6,7
  I_vv = None
  R_ijab += 0.5*np.einsum('ijkl,klab->ijab',Ioooo,t2)    #diagram 22 
  Ioooo = None
  R_ijab += 2*np.einsum('jcbk,kica->ijab',Iovvo,t2)      #diagram 19,28,20   
  Iovvo =None
  R_ijab += - np.einsum('jcbk,ikca->ijab',Iovvo_2,t2)    #diagram 21,29 
  Iovvo_2 = None
  R_ijab += - np.einsum('ickb,kjac->ijab',Iovov,t2)      #diagram 23 
  Iovov = None
  return R_ijab
  R_ijab = None
  gc.collect()

##--------Linear Response Theory-------##

def singles_n_doubles_response(t1,I_oovo,I_vovv):
  R_ijab = -np.einsum('ijak,kb->ijab',I_oovo,t1)       #diagrams 11,12,13,15,17
  I_oovo = None
  R_ijab += np.einsum('cjab,ic->ijab',I_vovv,t1)       #diagrams 9,10,14,16,18
  I_vovv = None
  return R_ijab
  R_ijab = None
  gc.collect() 

##------Explicit diagrams for iCCSDn scheme------##
 
def inserted_diag_So(t2,II_oo):
  R_ijab = -np.einsum('ik,kjab->ijab',II_oo,t2)   
  return R_ijab 

  R_ijab = None
  II_oo = None
  gc.collect()               ####WHERE X IS T2 OF ZEROTH OR FIRST ORDER AND Y IS THE CORRESPONDING INTERMEDIATE OF FIRST OR ZEROTH ORDER####

def inserted_diag_Sv(t2,II_vv):
  R_ijab = np.einsum('ca,ijcb->ijab',II_vv,t2)  
  return R_ijab

  R_ijab = None
  II_vv = None
  gc.collect()

##--------Constructing S diagrams-------------------##

def Sv_diagram_vs_contraction_response(Sv):
  R_iuab = -np.einsum('ik,kuab->iuab',Fock_mo[:occ,:occ],Sv)
  R_iuab += np.einsum('da,iudb->iuab',Fock_mo[occ:nao,occ:nao],Sv)
  R_iuab += np.einsum('db,iuad->iuab',Fock_mo[occ:nao,occ:nao],Sv)
  R_iuab += np.einsum('edab,iued->iuab',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],Sv)
  R_iuab += 2*np.einsum('idak,kudb->iuab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],Sv) 
  R_iuab += -np.einsum('idka,kudb->iuab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],Sv)
  R_iuab += -np.einsum('dika,kubd->iuab',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],Sv)
  R_iuab += -np.einsum('idkb,kuad->iuab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],Sv)
  #R_iuab += -np.einsum('iwab,uw->iuab',x,II_vv[:v_act,:v_act])     #### -(Sv(H,Sv)_c)_c term
  return R_iuab
  
  R_iuab = None
  II_vv = None   
  gc.collect()
  
  
def Sv_diagram_vt_contraction_response(t2):
  R_iuab = 2*np.einsum('dukb,kida->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],t2)
  R_iuab += -np.einsum('udkb,kida->iuab',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],t2)
  R_iuab += -np.einsum('dukb,kiad->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],t2)
  R_iuab += np.einsum('uikl,klba->iuab',twoelecint_mo[occ:occ+v_act,:occ,:occ,:occ],t2)
  R_iuab += -np.einsum('udka,kibd->iuab',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],t2)
  return R_iuab
  
  R_iuab = None
  gc.collect() 
 
def So_diagram_vs_contraction_response(So):
  R_ijav = np.einsum('da,ijdv->ijav',Fock_mo[occ:nao,occ:nao],So)
  R_ijav += -np.einsum('jl,ilav->ijav',Fock_mo[:occ,:occ],So)
  R_ijav += -np.einsum('il,ljav->ijav',Fock_mo[:occ,:occ],So)
  R_ijav += 2*np.einsum('dila,ljdv->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],So)
  R_ijav += -np.einsum('dila,jldv->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],So)   
  R_ijav += -np.einsum('dial,ljdv->ijav',twoelecint_mo[occ:nao,:occ,occ:nao,:occ],So)
  R_ijav += np.einsum('ijlm,lmav->ijav',twoelecint_mo[:occ,:occ,:occ,:occ],So)
  R_ijav += -np.einsum('jdla,ildv->ijav',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],So)
  #R_ijav += np.einsum('ijax,xv->ijav',x,II_oo[occ-o_act:occ,occ-o_act:occ])     #### -(So(H.So)_c)_c term
  return R_ijav
  
  R_ijav = None
  II_oo = None
  gc.collect()

def So_diagram_vt_contraction_response(t2):
  R_ijav = -np.einsum('djlv,liad->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],t2)
  R_ijav += -np.einsum('djvl,lida->ijav',twoelecint_mo[occ:nao,:occ,occ-o_act:occ,:occ],t2)
  R_ijav += np.einsum('cdva,jicd->ijav',twoelecint_mo[occ:nao,occ:nao,occ-o_act:occ,occ:nao],t2)
  R_ijav = -np.einsum('idlv,ljad->ijav',twoelecint_mo[:occ,occ:nao,:occ,occ-o_act:occ],t2)
  R_ijav += 2*np.einsum('djlv,lida->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],t2)
  return R_ijav

  R_ijav = None
  gc.collect()

def T1_contribution_Sv_response(t1):
  R_iuab = -np.einsum('uika,kb->iuab',twoelecint_mo[occ:occ+v_act,:occ,:occ,occ:nao],t1)
  R_iuab += np.einsum('duab,id->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,occ:nao,occ:nao],t1)
  R_iuab += -np.einsum('iukb,ka->iuab',twoelecint_mo[:occ,occ:occ+v_act,:occ,occ:nao],t1)
  return R_iuab
  R_iuab = None

def T1_contribution_So_response(t1):
  R_ijav = np.einsum('diva,jd->ijav',twoelecint_mo[occ:nao,:occ,occ-o_act:occ,occ:nao],t1)
  R_ijav += np.einsum('djav,id->ijav',twoelecint_mo[occ:nao,:occ,occ:nao,occ-o_act:occ],t1)
  R_ijav += -np.einsum('ijkv,ka->ijav',twoelecint_mo[:occ,:occ,:occ,occ-o_act:occ],t1)
  return R_ijav
  R_ijav = None

