# Import modules
import gc
import numpy as np
import copy as cp
import trans_mo
import MP2
import inp
import intermediates
import trans_mo

# import important stuff
nao = MP2.nao
twoelecint_mo = MP2.twoelecint_mo 
Fock_mo = MP2.Fock_mo
#Fock_mo = trans_mo.oneelecint_mo
D1 = MP2.D1
D2=MP2.D2
So = MP2.So
Do=MP2.Do
Sv = MP2.Sv
Dv=MP2.Dv
occ = MP2.occ
virt = MP2.virt
conv = 10**(-inp.conv)

#      active orbital numbers imported from input file
o_act = inp.o_act
v_act = inp.v_act
act = o_act + v_act

def zero(ehf,t0,t1,t2):
  
  R_0 = ehf*t0
  R_0 += 2.0*np.einsum('ia,ia',Fock_mo[:occ,occ:nao],t1)
  R_0 += 2.0*np.einsum('ijab,ijab',twoelecint_mo[:occ,:occ,occ:nao,occ:nao],t2)
  R_0 += -1.0*np.einsum('ijab,ijba',twoelecint_mo[:occ,:occ,occ:nao,occ:nao],t2)

  return R_0
  R_0 = None
  gc.collect()

#      Compute R_ia
def singles(I1,I2,I_oo,I_vv,tau,t0,t1,t2,ehf):

  R_ia = Fock_mo[:occ,occ:nao]*t0
  R_ia += ehf*t1
  R_ia += -np.einsum('ik,ka->ia',I_oo,t1)                                          #diagrams 1,l,j,m,n
  I_oo = None
  R_ia += np.einsum('ca,ic->ia',I_vv,t1)                                           #diagrams 2,k,i
  I_vv = None
  R_ia += -2*np.einsum('ibkj,kjab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)     #diagrams 5 and a
  R_ia += np.einsum('ibkj,jkab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)     #diagrams 6 and b
  R_ia += 2*np.einsum('cdak,ikcd->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2) #diagrams 7 and c
  R_ia += -np.einsum('cdak,ikdc->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2) #diagrams 8 and d
  R_ia += 2*np.einsum('icak,kc->ia',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1)           #diagram 3
  R_ia += -np.einsum('icka,kc->ia',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1)           #diagram 4

  return R_ia
  R_ia = None
  gc.collect()

#      Compute R_ijab
def doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,I_oovo,I_vovv,t0,t1, tau,t2,ehf):
  print " "

  R_ijab = 0.5*cp.deepcopy(twoelecint_mo[:occ,:occ,occ:nao,occ:nao])*t0
  R_ijab += 0.5*ehf*t2
  R_ijab += -np.einsum('ik,kjab->ijab',I_oo,t2)            #diagrams 1,25,27,5,8,35,38'
  I_oo = None
  R_ijab += np.einsum('ca,ijcb->ijab',I_vv,t2)              #diagrams 2,24,25,34',non-linear 6,7
  I_vv = None
  R_ijab += 0.5*np.einsum('cdab,ijcd->ijab',Ivvvv,tau)      #diagram 2,linear 5 # changed the factor from 0.5 to 1.0
  Ivvvv = None
  R_ijab += 0.5*np.einsum('ijkl,klab->ijab',Ioooo,tau)          #diagrams 1,38 and linear 9 with twoelecint # changed the factor from 0.5 to 1.0
  Ioooo = None
  R_ijab += 2.0*np.einsum('jcbk,kica->ijab',Iovvo,t2)      #diagram 6 linear if we use twoelecint 
  Iovvo =None
  R_ijab += - np.einsum('jcbk,ikca->ijab',Iovvo_2,t2)  #diagram 8 linear if we use twoelecint 
  Iovvo_2 = None
  R_ijab += - np.einsum('ickb,kjac->ijab',Iovov,t2)    #linear 10 with twoelecint
  Iovov = None
  R_ijab += -np.einsum('icka,kjcb->ijab',Iovov_2,t2)  #linear 7 with twoelecint
  Iovov_2 = None
  R_ijab += -np.einsum('ijak,kb->ijab',I_oovo,t1)       #diagrams 11,12,13,15,17
  Ioovo = None
  R_ijab += np.einsum('cjab,ic->ijab',I_vovv,t1)       #diagrams 9,10,14,16,18,31,37,39
  Ivovovvv = None


  return R_ijab
  R_ijab = None
  gc.collect()

