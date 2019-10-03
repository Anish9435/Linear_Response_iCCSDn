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
conv = 10**(-inp.conv)

##-------active orbital numbers imported from input file--------##
o_act = inp.o_act
v_act = inp.v_act
act = o_act + v_act

##-------t1 and t2 contributing to R_ia-------##

def singles(I1,I2,I_oo,I_vv,tau,t1,t2):
  R_ia = cp.deepcopy(Fock_mo[:occ,occ:nao])
  R_ia += -np.einsum('ik,ka->ia',I_oo,t1)                                          #diagrams 1,l,j,m,n
  I_oo = None
  R_ia += np.einsum('ca,ic->ia',I_vv,t1)                                           #diagrams 2,k,i
  I_vv = None
  R_ia += -2*np.einsum('ibkj,kjab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],tau)     #diagrams 5 and a
  R_ia += np.einsum('ibkj,jkab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],tau)     #diagrams 6 and b
  R_ia += 2*np.einsum('cdak,ikcd->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],tau) #diagrams 7 and c
  R_ia += -np.einsum('cdak,ikdc->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],tau) #diagrams 8 and d
  

  R_ia += 2*np.einsum('bj,ijab->ia',I1,t2) - np.einsum('bj,ijba->ia',I1,t2)     #diagrams e,f
  I1 = None
  R_ia += 2*np.einsum('bj,ijab->ia',I2,t2) - np.einsum('bj,ijba->ia',I2,t2)     #diagrams g,h
  I2 = None
  R_ia += 2*np.einsum('icak,kc->ia',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1)           #diagram 3
  R_ia += -np.einsum('icka,kc->ia',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1)           #diagram 4
  return R_ia
  R_ia = None
  gc.collect()


##-------t2 and tau contributing to R_ijab-----------##

def doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,t2):
  print " "
  R_ijab = 0.5*cp.deepcopy(twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
  R_ijab += -np.einsum('ik,kjab->ijab',I_oo,t2)        #diagrams linear 1 and non-linear 25,27,5,8,35,38'
  I_oo = None
  R_ijab += np.einsum('ca,ijcb->ijab',I_vv,t2)         #diagrams linear 2 and non-linear 24,26,34',6,7
  I_vv = None
  R_ijab += 0.5*np.einsum('cdab,ijcd->ijab',Ivvvv,tau) #diagrams linear 5 and non-linear 2
  Ivvvv = None
  R_ijab += 0.5*np.einsum('ijkl,klab->ijab',Ioooo,t2)  #diagrams linear 9 and non-linear 22,38
  Ioooo = None
  R_ijab += 2*np.einsum('jcbk,kica->ijab',Iovvo,t2)    #diagrams linear 6 and non-linear 19,28,20
  Iovvo =None
  R_ijab += - np.einsum('jcbk,ikca->ijab',Iovvo_2,t2)  #diagrams linear 8 and non-linear 21,29 
  Iovvo_2 = None
  R_ijab += - np.einsum('ickb,kjac->ijab',Iovov,t2)    #diagrams linear 10 and non-linear 23
  Iovov = None
  R_ijab += -np.einsum('icka,kjcb->ijab',Iovov_2,t2)   #diagram linear 7
  Iovov_2 = None
  return R_ijab
  R_ijab = None
  gc.collect()


##------t1 terms contributing to R_ijab--------##

def singles_n_doubles(t1,I_oovo,I_vovv):
  R_ijab = -np.einsum('ijak,kb->ijab',I_oovo,t1)       #diagrams 11,12,13,15,17
  I_oovo = None
  R_ijab += np.einsum('cjab,ic->ijab',I_vovv,t1)       #diagrams 9,10,14,16,18
  I_vovv = None
  R_ijab += -np.einsum('ijkb,ka->ijab',twoelecint_mo[:occ,:occ,:occ,occ:nao],t1)            #diagram 3
  R_ijab += np.einsum('cjab,ic->ijab',twoelecint_mo[occ:nao,:occ,occ:nao,occ:nao],t1)       #diagram 4
  R_ijab += -np.einsum('ickb,ka,jc->ijab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,t1)   #diagrams non-linear 3
  R_ijab += -np.einsum('icak,jc,kb->ijab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,t1)   #diagrams non-linear 4
  R_ijab += 0.5*np.einsum('ijkl,ka,lb->ijab',twoelecint_mo[:occ,:occ,:occ,:occ],t1,t1)      #diagram non-linear 1
  return R_ijab
  R_ijab = None
  gc.collect() 


##------Higher orders of t1 contributing to R_ijab-------##

def higher_order(t1,t2,II_d,II_e):#,Iooov,I3,Ioooo_2,I_voov,II_d,II_e):
  R_ijab = -np.einsum('ijkb,ka->ijab',II_d,t1)       #diagrams 36
  R_ijab += -np.einsum('jibk,ka->ijab',II_e,t1)      #diagrams 32,33,31,30
  R_ijab += -np.einsum('ijlb,la->ijab',Iooov,t1)      #diagram 34,30
  R_ijab += -0.5*np.einsum('idal,jd,lb->ijab',I3,t1,t1)      #diagram 40
  R_ijab += np.einsum('ijkl,klab->ijab',Ioooo_2,t2)      #diagram 37
  R_ijab += -np.einsum('cjlb,ic,la->ijab',I_voov,t1,t1)      #diagram 39
  return R_ijab
  R_ijab = None 
  Iovvo_3 = None 
  Iooov = None 
  I3 = None 
  Ioooo_2 = None 
  I_voov = None
  II_d = None
  gc.collect()

##--------Constructing S diagrams-------------------##

def Sv_diagram_vs_contraction(Sv,II_vv):
  R_iuab = cp.deepcopy(twoelecint_mo[:occ,occ:occ+v_act,occ:nao,occ:nao])
  R_iuab += -np.einsum('ik,kuab->iuab',Fock_mo[:occ,:occ],Sv)
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
  
  
def Sv_diagram_vt_contraction(t2):
  R_iuab = 2*np.einsum('dukb,kida->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],t2)
  R_iuab += -np.einsum('udkb,kida->iuab',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],t2)
  R_iuab += -np.einsum('dukb,kiad->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],t2)
  R_iuab += np.einsum('uikl,klba->iuab',twoelecint_mo[occ:occ+v_act,:occ,:occ,:occ],t2)
  R_iuab += -np.einsum('udka,kibd->iuab',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],t2)
  return R_iuab
  
  R_iuab = None
  gc.collect() 
 
def So_diagram_vs_contraction(So,II_oo):
  R_ijav = cp.deepcopy(twoelecint_mo[:occ,:occ,occ:nao,occ-o_act:occ])
  R_ijav += np.einsum('da,ijdv->ijav',Fock_mo[occ:nao,occ:nao],So)
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

def So_diagram_vt_contraction(t2):
  R_ijav = -np.einsum('djlv,liad->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],t2)
  R_ijav += -np.einsum('djvl,lida->ijav',twoelecint_mo[occ:nao,:occ,occ-o_act:occ,:occ],t2)
  R_ijav += np.einsum('cdva,jicd->ijav',twoelecint_mo[occ:nao,occ:nao,occ-o_act:occ,occ:nao],t2)
  R_ijav += -np.einsum('idlv,ljad->ijav',twoelecint_mo[:occ,occ:nao,:occ,occ-o_act:occ],t2)
  R_ijav += 2*np.einsum('djlv,lida->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],t2)
  return R_ijav

  R_ijav = None
  gc.collect()

##---------T1 contribution to R_iuab and R_ijav---------##

def T1_contribution_Sv(t1):
  R_iuab = -np.einsum('uika,kb->iuab',twoelecint_mo[occ:occ+v_act,:occ,:occ,occ:nao],t1)
  R_iuab += np.einsum('duab,id->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,occ:nao,occ:nao],t1)
  R_iuab += -np.einsum('iukb,ka->iuab',twoelecint_mo[:occ,occ:occ+v_act,:occ,occ:nao],t1)
  return R_iuab
  R_iuab = None

def T1_contribution_So(t1):
  R_ijav = np.einsum('diva,jd->ijav',twoelecint_mo[occ:nao,:occ,occ-o_act:occ,occ:nao],t1)
  R_ijav += np.einsum('djav,id->ijav',twoelecint_mo[occ:nao,:occ,occ:nao,occ-o_act:occ],t1)
  R_ijav += -np.einsum('ijkv,ka->ijav',twoelecint_mo[:occ,:occ,:occ,occ-o_act:occ],t1)
  return R_ijav
  R_ijav = None

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