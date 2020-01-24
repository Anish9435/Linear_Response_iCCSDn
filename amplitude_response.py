
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                                              # Routine to calculate the guess to determine Excitation Energy #
                                    
                                   # Calculate the different t and s diagrams associated with the Linear Response Theory #

                                           # Author: Anish Chakraborty, Pradipta Samanta & Rahul Maitra #
                                                           # Date - 10th Dec, 2019 # 

                   ##----------------------------------------------------------------------------------------------------------------##

##------------------------------------------------------##
                 #Import modules#
##------------------------------------------------------##

import gc
import numpy as np
import copy as cp
import trans_mo
import MP2
import inp
import intermediates

##--------------------------------------------------------##
              #Import important variables#
##--------------------------------------------------------##

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


##-----------------------------------------------------------------------------------------------##
                #Function exclusive for LRT i.e there is no bare hamiltonian term#
##-----------------------------------------------------------------------------------------------##

def singles_response_linear(I_oo,I_vv,t1,t2):
  R_ia = -np.einsum('ik,ka->ia',I_oo,t1)   #diagrams 1
  R_ia += np.einsum('ca,ic->ia',I_vv,t1)   #diagrams 2
  R_ia += 2*np.einsum('icak,kc->ia',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1)    #diagram 3
  R_ia += -np.einsum('icka,kc->ia',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1)     #diagram 4
  R_ia += -2*np.einsum('ibkj,kjab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)     #diagrams 5 
  R_ia += np.einsum('ibkj,jkab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)     #diagrams 6 
  R_ia += 2*np.einsum('cdak,ikcd->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2) #diagrams 7 
  R_ia += -np.einsum('cdak,ikdc->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2) #diagrams 8 
  return R_ia

  R_ia = None
  I_oo = None
  I_vv = None
  gc.collect()

##-------------------------------------------------------------------------------------------##
                                    #Quardratic terms#
##-------------------------------------------------------------------------------------------##

def singles_response_quadratic(I_oo,I_vv,I1,I2,t1,t2):
  R_ia = -np.einsum('ik,ka->ia',I_oo,t1)   #diagrams l,j
  R_ia += np.einsum('ca,ic->ia',I_vv,t1)   #diagrams k,i
  R_ia += 2*np.einsum('bj,ijab->ia',I1,t2) - np.einsum('bj,ijba->ia',I1,t2)     #diagrams e,f
  R_ia += 2*np.einsum('bj,ijab->ia',I2,t2) - np.einsum('bj,ijba->ia',I2,t2)     #diagrams g,h
  return R_ia

  R_ia = None
  I_oo = None
  I_vv = None
  I1 = None
  I2 = None
  gc.collect()

##------------------------------------------------------------------------------------------------##
                   #Similar as R_ia for LRT i.e no bare hamiltonian term#
##------------------------------------------------------------------------------------------------##

def doubles_response_linear(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,t1,t2):
  R_ijab = -np.einsum('ik,kjab->ijab',I_oo,t2)  #diagram 1
  R_ijab += np.einsum('ca,ijcb->ijab',I_vv,t2)  #diagram 2
  R_ijab += -np.einsum('ijkb,ka->ijab',twoelecint_mo[:occ,:occ,:occ,occ:nao],t1)      #diagram 3
  R_ijab += np.einsum('cjab,ic->ijab',twoelecint_mo[occ:nao,:occ,occ:nao,occ:nao],t1) #diagram 4
  R_ijab += 0.5*np.einsum('cdab,ijcd->ijab',Ivvvv,t2)  #diagram 5
  R_ijab += 2*np.einsum('jcbk,kica->ijab',Iovvo,t2)      #diagram 6 
  R_ijab += -np.einsum('icka,kjcb->ijab',Iovov_2,t2)  #diagram 7 
  R_ijab += -np.einsum('jcbk,ikca->ijab',Iovvo_2,t2)  #diagram 8  
  R_ijab += 0.5*np.einsum('ijkl,klab->ijab',Ioooo,t2)  #diagram 9 
  R_ijab += -np.einsum('ickb,kjac->ijab',Iovov,t2)    #linear 10 
  return R_ijab

  R_ijab = None
  I_oo = None
  I_vv = None
  Ivvvv = None
  Iovvo =None
  Iovov_2 = None
  Iovvo_2 = None
  Ioooo = None
  Iovov = None
  gc.collect()

##-------------------------------------------------------------------------------------##
                              #Quadratic terms#
##-------------------------------------------------------------------------------------##

def doubles_response_quadratic(I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov,t2):
  R_ijab = -np.einsum('ik,kjab->ijab',I_oo,t2)     #diagrams 25,27,5,8
  R_ijab += np.einsum('ca,ijcb->ijab',I_vv,t2)     #diagrams 24,26,6,7
  R_ijab += 0.5*np.einsum('ijkl,klab->ijab',Ioooo,t2)    #diagram 22 
  R_ijab += 2*np.einsum('jcbk,kica->ijab',Iovvo,t2)      #diagram 19,28,20   
  R_ijab += - np.einsum('jcbk,ikca->ijab',Iovvo_2,t2)    #diagram 21,29 
  R_ijab += - np.einsum('ickb,kjac->ijab',Iovov,t2)      #diagram 23 
  return R_ijab

  R_ijab = None
  I_oo = None
  I_vv = None
  Ioooo = None
  Iovvo =None
  Iovvo_2 = None
  Iovov = None
  gc.collect()

##-----------------------------------------------------------------------------------##
                              #Linear Response Theory#
##-----------------------------------------------------------------------------------##

def singles_n_doubles_response(t1,I_oovo,I_vovv):
  R_ijab = -np.einsum('ijak,kb->ijab',I_oovo,t1)       #diagrams 11,12,13,15,17
  R_ijab += np.einsum('cjab,ic->ijab',I_vovv,t1)       #diagrams 9,10,14,16,18
  return R_ijab

  R_ijab = None
  I_oovo = None
  I_vovv = None
  gc.collect() 

##-----------------------------------------------------------------------------------##
                        #Explicit diagrams for iCCSDn scheme#
                      #(ST)_c leading to the triple excitation#
##-----------------------------------------------------------------------------------##

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

##-----------------------------------------------------------------------------------##
                        #Explicit diagrams for iCCSDn scheme#
                      #(ST)_c leading to the triple excitation#
##-----------------------------------------------------------------------------------##

def inserted_diag_So_t1(t1,II_oo):
  R_ia = -np.einsum('ik,ka->ia',II_oo,t1)   
  return R_ia 

  R_ia = None
  II_oo = None
  gc.collect()               

def inserted_diag_Sv_t1(t1,II_vv):
  R_ia = np.einsum('ca,ic->ia',II_vv,t1)  
  return R_ia

  R_ia = None
  II_vv = None
  gc.collect()

##-----------------------------------------------------------------------------------##
                        #(vst)_c terms contributing towards#
                                  #R_iuab and R_ijav#
##-----------------------------------------------------------------------------------##
 
def v_so_t_contraction_diag(t2,II_ov):
  R_iuab = -np.einsum('ux,xiba->iuab',II_ov,t2) 
  return R_iuab

  R_iuab = None
  II_ov = None
  gc.collect()

def v_sv_t_contraction_diag(t2,II_vo):
  R_ijav = np.einsum('wv,jiwa->ijav',II_vo,t2)
  return R_ijav

  R_ijav = None
  II_vo = None
  gc.collect()

##----------------------------------------------------------------------------------------------##
             #Constructing S diagrams (i.e VS_v contraction contributing to R_iuab)#
##----------------------------------------------------------------------------------------------##

def Sv_diagram_vs_contraction_response(Sv):
  R_iuab = -np.einsum('ik,kuab->iuab',Fock_mo[:occ,:occ],Sv)
  R_iuab += np.einsum('da,iudb->iuab',Fock_mo[occ:nao,occ:nao],Sv)
  R_iuab += np.einsum('db,iuad->iuab',Fock_mo[occ:nao,occ:nao],Sv)
  R_iuab += np.einsum('edab,iued->iuab',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],Sv)
  R_iuab += 2*np.einsum('idak,kudb->iuab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],Sv) 
  R_iuab += -np.einsum('idka,kudb->iuab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],Sv)
  R_iuab += -np.einsum('dika,kubd->iuab',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],Sv)
  R_iuab += -np.einsum('idkb,kuad->iuab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],Sv)
  return R_iuab
  
  R_iuab = None
  II_vv = None   
  gc.collect()
  
##----------------------------------------------------------------------------------------------##
             #Constructing S diagrams (i.e Vt_2 contraction contributing to R_iuab)#
##----------------------------------------------------------------------------------------------##
  
def Sv_diagram_vt_contraction_response(t2):
  R_iuab = 2*np.einsum('dukb,kida->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],t2)
  R_iuab += -np.einsum('udkb,kida->iuab',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],t2)
  R_iuab += -np.einsum('dukb,kiad->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],t2)
  R_iuab += np.einsum('uikl,klba->iuab',twoelecint_mo[occ:occ+v_act,:occ,:occ,:occ],t2)
  R_iuab += -np.einsum('udka,kibd->iuab',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],t2)
  return R_iuab
  
  R_iuab = None
  gc.collect() 
 
##----------------------------------------------------------------------------------------------##
             #Constructing S diagrams (i.e VS_o contraction contributing to R_ijav)#
##----------------------------------------------------------------------------------------------##

def So_diagram_vs_contraction_response(So):
  R_ijav = np.einsum('da,ijdv->ijav',Fock_mo[occ:nao,occ:nao],So)
  R_ijav += -np.einsum('jl,ilav->ijav',Fock_mo[:occ,:occ],So)
  R_ijav += -np.einsum('il,ljav->ijav',Fock_mo[:occ,:occ],So)
  R_ijav += 2*np.einsum('dila,ljdv->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],So)
  R_ijav += -np.einsum('dila,jldv->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],So)   
  R_ijav += -np.einsum('dial,ljdv->ijav',twoelecint_mo[occ:nao,:occ,occ:nao,:occ],So)
  R_ijav += np.einsum('ijlm,lmav->ijav',twoelecint_mo[:occ,:occ,:occ,:occ],So)
  R_ijav += -np.einsum('jdla,ildv->ijav',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],So)
  return R_ijav
  
  R_ijav = None
  II_oo = None
  gc.collect()

##----------------------------------------------------------------------------------------------##
             #Constructing S diagrams (i.e Vt_2 contraction contributing to R_ijav)#
##----------------------------------------------------------------------------------------------##

def So_diagram_vt_contraction_response(t2):
  R_ijav = -np.einsum('djlv,liad->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],t2)
  R_ijav += -np.einsum('djvl,lida->ijav',twoelecint_mo[occ:nao,:occ,occ-o_act:occ,:occ],t2)
  R_ijav += np.einsum('cdva,jicd->ijav',twoelecint_mo[occ:nao,occ:nao,occ-o_act:occ,occ:nao],t2)
  R_ijav += -np.einsum('idlv,ljad->ijav',twoelecint_mo[:occ,occ:nao,:occ,occ-o_act:occ],t2)
  R_ijav += 2*np.einsum('djlv,lida->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],t2)
  return R_ijav

  R_ijav = None
  gc.collect()

##----------------------------------------------------------------------------------------------##
             #Constructing S diagrams (i.e Vt_1 contraction contributing to R_iuab)#
##----------------------------------------------------------------------------------------------##

def T1_contribution_Sv_response(t1):
  R_iuab = -np.einsum('uika,kb->iuab',twoelecint_mo[occ:occ+v_act,:occ,:occ,occ:nao],t1)
  R_iuab += np.einsum('duab,id->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,occ:nao,occ:nao],t1)
  R_iuab += -np.einsum('iukb,ka->iuab',twoelecint_mo[:occ,occ:occ+v_act,:occ,occ:nao],t1)
  return R_iuab

  R_iuab = None
  gc.collect()

##----------------------------------------------------------------------------------------------##
             #Constructing S diagrams (i.e Vt_1 contraction contributing to R_ijav)#
##----------------------------------------------------------------------------------------------##

def T1_contribution_So_response(t1):
  R_ijav = np.einsum('diva,jd->ijav',twoelecint_mo[occ:nao,:occ,occ-o_act:occ,occ:nao],t1)
  R_ijav += np.einsum('djav,id->ijav',twoelecint_mo[occ:nao,:occ,occ:nao,occ-o_act:occ],t1)
  R_ijav += -np.einsum('ijkv,ka->ijav',twoelecint_mo[:occ,:occ,:occ,occ-o_act:occ],t1)
  return R_ijav

  R_ijav = None
  gc.collect()

##----------------------------------------------------------------------------------------------##
                      #Non linear response of S towards R_iuab and R_ijav#
##----------------------------------------------------------------------------------------------##

def nonlinear_So_response(II_oo,So):
  R_ijav = np.einsum('ijax,xv->ijav',So,II_oo[occ-o_act:occ,occ-o_act:occ])     #### -(So(H.So)_c)_c term
  return R_ijav               ###additional -ve for -(So(H.So)_c)_c term 

  R_ijav = None
  gc.collect()

def nonlinear_Sv_response(II_vv,Sv):
  R_iuab = -np.einsum('iwab,uw->iuab',Sv,II_vv[:v_act,:v_act])     #### -(Sv(H,Sv)_c)_c term
  return R_iuab            ###additional -ve for -(Sv(H.Sv)_c)_c term

  R_iuab = None
  gc.collect()

##-----------------------------------------------------------------------------------##
                        #Two body (vst)_c terms contributing towards#
                                  #R_iuab and R_ijav#
##-----------------------------------------------------------------------------------##

def w2_diag_So_response(II_ovoo,II_vvvo2,II_ovoo2,t2):
  R_ijav = 2.0*np.einsum('jdvw,wida->ijav',II_ovoo,t2)
  R_ijav += -np.einsum('jdvw,wiad->ijav',II_ovoo,t2) #diagonal terms
  R_ijav += np.einsum('dxav,ijdx->ijav',II_vvvo2,t2) #off-diagonal terms
  R_ijav += -np.einsum('ixkv,kjax->ijav',II_ovoo2,t2)
  R_ijav += -np.einsum('jxkv,kixa->ijav',II_ovoo2,t2)
  return R_ijav

  R_ijav = None
  II_ovoo = None
  II_vvvo2 = None
  II_ovoo2 = None
  gc.collect()

def w2_diag_Sv_response(II_vvvo,II_ovoo3,II_vvvo3,t2):
  R_iuab = 2.0*np.einsum('uxbl,ilax->iuab',II_vvvo,t2)
  R_iuab += -np.einsum('uxbl,ilxa->iuab',II_vvvo,t2)
  R_iuab += -np.einsum('iulw,lwab->iuab',II_ovoo3,t2)
  R_iuab += -np.einsum('duaw,iwdb->iuab',II_vvvo3,t2) 
  R_iuab += -np.einsum('dubw,iwad->iuab',II_vvvo3,t2) 
  return R_iuab

  R_iuab = None
  II_vvvo = None
  II_ovoo3 = None
  II_vvvo3 = None
  gc.collect()


                  ##-----------------------------------------------------------------------------------------------------------------------##
                                                                         #THE END#
                  ##-----------------------------------------------------------------------------------------------------------------------##
