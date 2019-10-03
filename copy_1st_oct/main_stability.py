# Import modules
import numpy as np
import copy as cp
import trans_mo
import MP2
import inp
import intermediates
import amplitude
import cc_update
import cc_symmetrize
from opt_einsum import contract
import main

mol = inp.mol

# Obtain the number of atomic orbitals in the basis set
nao = main.nao

# import important stuff
Fock_mo = MP2.Fock_mo
twoelecint_mo = MP2.twoelecint_mo
hf_mo_E = MP2.hf_mo_E
t1 = main.t1
t2 = main.t2
occ = MP2.occ
virt = MP2.virt
red_occ = (occ*(occ+1))/2
red_virt = (virt*(virt+1))/2

##-----active orbitals numbers------##
o_act = inp.o_act
v_act = inp.v_act
act = o_act + v_act
##---------------------------------##
##------Delta function-------------##
o_delta = np.zeros((occ,occ))
for i in range(0,occ):
  o_delta[i,i] = 1.0

v_delta = np.zeros((virt,virt)) 
for a in range(0,virt):
  v_delta[a,a] = 1.0
##---------------------------------##

############t1_diagrams#################
#differentiation w.r.t t1(p,r)
##------------------------Linear terms------------------------------------##
J11 = -np.einsum('ik,kp,ra->iapr',Fock_mo[:occ,:occ],o_delta,v_delta) #diag 1
J11 += np.einsum('ca,ip,rc->iapr',Fock_mo[occ:nao,occ:nao],o_delta,v_delta) #diag 2
J11 += 2*contract('icak,kp,rc->iapr',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],o_delta,v_delta) #diag 3
J11 += -contract('icka,kp,rc->iapr',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],o_delta,v_delta) #diag 4
##---------------------------------------------------------------------##
'''
##-----------------------Non-linear terms----------------------------##
J11 += -2*contract('ibkj,ka,jp,br->iapr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,v_delta) #diag a
J11 += -2*contract('ibkj,jb,kp,ar->iapr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,v_delta) #diag a
J11 += contract('ibkj,ja,kp,br->iapr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,v_delta) #diag b
J11 += contract('ibkj,kb,jp,ar->iapr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,v_delta) #diag b
J11 += 2*contract('cbaj,jb,ip,cr->iapr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,v_delta) #diag c
J11 += 2*contract('cbaj,ic,jp,br->iapr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,v_delta) #diag c
J11 += -contract('cdak,id,kp,cr->iapr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,v_delta) #diag d
J11 += -contract('cdak,kc,ip,dr->iapr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,v_delta) #diag d
J11 += 4*contract('cbkj,ijab,kp,cr->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag e
J11 += -2*contract('cbkj,ijba,kp,cr->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag f
J11 += -2*contract('cbjk,ijab,kp,cr->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag g
J11 += contract('cbjk,ijba,kp,cr->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag h
J11 += contract('cdkl,klda,ip,cr->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag i
J11 += contract('dckl,lidc,kp,ar->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag j
J11 += -2*contract('cdkl,klad,ip,cr->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag k
J11 += -2*contract('cdkl,ilcd,kp,ar->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag l
##---------------------------------------------------------------------##
##----------------------Higher order terms-----------------------------##
J11 += -2*contract('cdkl,ld,ic,kp,ar->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag n
J11 += -2*contract('cdkl,ld,ka,ip,cr->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag n
J11 += -2*contract('cdkl,ic,ka,lp,dr->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag n
J11 += contract('cdlk,ic,ld,kp,ar->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag m
J11 += contract('cdlk,ic,ka,lp,dr->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag m
J11 += contract('cdlk,ld,ka,ip,cr->iapr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag m
##---------------------------------------------------------------------##
'''
a = np.reshape(J11,(occ*virt,occ*virt))
#print 'J11',J11.shape,a.shape


#differentiation w.r.t t2(p,q,r,s)
##---------------Linear terms-----------------------------------##
J12 = -2*contract('ibkj,kp,jq,ar,bs->iapqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],o_delta,o_delta,v_delta,v_delta) #diag 5
J12 += contract('ibkj,jp,kq,ar,bs->iapqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],o_delta,o_delta,v_delta,v_delta) #diag 6
J12 += 2*contract('cdak,ip,kq,cr,ds->iapqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],o_delta,o_delta,v_delta,v_delta) #diag 7
J12 += -contract('cdak,ip,kq,ds,cr->iapqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],o_delta,o_delta,v_delta,v_delta) #diag 8
##--------------------------------------------------------------##
'''
##-------------Non-linear terms--------------------------##
J12 += 4*contract('cbkj,kc,ip,jq,ar,bs->iapqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag e
J12 += -2*contract('cbkj,kc,ip,jq,br,as->iapqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag f
J12 += -2*contract('cbjk,kc,ip,jq,ar,bs->iapqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag g
J12 += contract('cbjk,kc,ip,jq,br,as->iapqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag h
J12 += contract('cdkl,ic,kp,lq,dr,as->iapqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag i
J12 += contract('dckl,ka,lp,iq,dr,cs->iapqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag j
J12 += -2*contract('cdkl,ic,kp,lq,ar,ds->iapqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag k
J12 += -2*contract('cdkl,ka,ip,lq,cr,ds->iapqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) # diag l
##---------------------------------------------------------##
'''
b = np.reshape(J12,(occ*virt,occ*occ*virt*virt))
#print 'J12',J12.shape ,b.shape



#############t2_diagrams################# 
#differentiation w.r.t t1(p,r)
##----------------Linear terms----------------------------------##
J21 = -contract('ijkb,kp,ar->ijabpr',twoelecint_mo[:occ,:occ,:occ,occ:nao],o_delta,v_delta) #diag 3
J21 += -contract('jika,kp,br->ijabpr',twoelecint_mo[:occ,:occ,:occ,occ:nao],o_delta,v_delta) #diag 3
J21 += contract('cjab,ip,cr->ijabpr',twoelecint_mo[occ:nao,:occ,occ:nao,occ:nao],o_delta,v_delta) #diag 4
J21 += contract('ciba,jp,cr->ijabpr',twoelecint_mo[occ:nao,:occ,occ:nao,occ:nao],o_delta,v_delta) #diag 4
##---------------------------------------------------------##
'''
##-------------Non-linear terms---------------------------##
J21 += 0.5*contract('ijkl,ka,lp,br->ijabpr',twoelecint_mo[:occ,:occ,:occ,:occ],t1,o_delta,v_delta) #diag 1
J21 += 0.5*contract('jikl,kb,lp,ar->ijabpr',twoelecint_mo[:occ,:occ,:occ,:occ],t1,o_delta,v_delta) #diag 1
J21 += 0.5*contract('ijkl,lb,kp,ar->ijabpr',twoelecint_mo[:occ,:occ,:occ,:occ],t1,o_delta,v_delta) #diag 1
J21 += 0.5*contract('jikl,la,kp,br->ijabpr',twoelecint_mo[:occ,:occ,:occ,:occ],t1,o_delta,v_delta) #diag 1
J21 += 0.5*contract('cdab,ic,jp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],t1,o_delta,v_delta) #diag 2
J21 += 0.5*contract('cdba,jc,ip,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],t1,o_delta,v_delta) #diag 2
J21 += 0.5*contract('cdab,jd,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],t1,o_delta,v_delta) #diag 2
J21 += 0.5*contract('cdba,id,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],t1,o_delta,v_delta) #diag 2
J21 += -contract('ickb,ka,jp,cr->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,o_delta,v_delta) #diag 3
J21 += -contract('jcka,kb,ip,cr->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,o_delta,v_delta) #diag 3
J21 += -contract('ickb,jc,kp,ar->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,o_delta,v_delta) #diag 3
J21 += -contract('jcka,ic,kp,br->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,o_delta,v_delta) #diag 3
J21 += -contract('icak,jc,kp,br->ijabpr',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,o_delta,v_delta)   #diag 4
J21 += -contract('jcbk,ic,kp,ar->ijabpr',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,o_delta,v_delta)   #diag 4
J21 += -contract('icak,kb,jp,cr->ijabpr',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,o_delta,v_delta)   #diag 4
J21 += -contract('jcbk,ka,ip,cr->ijabpr',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,o_delta,v_delta)   #diag 4
J21 += -2*contract('idkl,kjab,lp,dr->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag 5
J21 += -2*contract('jdkl,kiba,lp,dr->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag 5
J21 += 2*contract('dcla,ijcb,lp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2,o_delta,v_delta) #diag 6
J21 += 2*contract('dclb,jica,lp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2,o_delta,v_delta) #diag 6
J21 += -contract('cdak,ijdb,kp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2,o_delta,v_delta) #diag 7
J21 += -contract('cdbk,jida,kp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2,o_delta,v_delta) #diag 7
J21 += contract('idkl,ljab,kp,dr->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag 8
J21 += contract('jdkl,liba,kp,dr->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag 8
J21 += contract('cjkl,klab,ip,cr->ijabpr',twoelecint_mo[occ:nao,:occ,:occ,:occ],t2,o_delta,v_delta)  #diag 9
J21 += contract('cikl,klba,jp,cr->ijabpr',twoelecint_mo[occ:nao,:occ,:occ,:occ],t2,o_delta,v_delta)  #diag 9
J21 += -contract('cdlb,ljad,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2,o_delta,v_delta) #diag 10
J21 += -contract('cdla,libd,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2,o_delta,v_delta) #diag 10
J21 += contract('cikl,jlca,kp,br->ijabpr',twoelecint_mo[occ:nao,:occ,:occ,:occ],t2,o_delta,v_delta) ##diag 11
J21 += contract('cjkl,ilcb,kp,ar->ijabpr',twoelecint_mo[occ:nao,:occ,:occ,:occ],t2,o_delta,v_delta) ##diag 11
J21 += -contract('cdka,jicd,kp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2,o_delta,v_delta) #diag 12
J21 += -contract('cdkb,ijcd,kp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2,o_delta,v_delta) #diag 12
J21 += contract('jclk,lica,kp,br->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag 13
J21 += contract('iclk,ljcb,kp,ar->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag 13
J21 += -contract('cdka,kjdb,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2,o_delta,v_delta) #diag 14
J21 += -contract('cdkb,kida,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2,o_delta,v_delta) #diag 14
J21 += -2*contract('jckl,lica,kp,br->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag 15
J21 += -2*contract('ickl,ljcb,kp,ar->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag 15
J21 += 2*contract('cdal,ljdb,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2,o_delta,v_delta) #diag 16
J21 += 2*contract('cdbl,lida,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2,o_delta,v_delta) #diag 16
J21 += contract('jckl,ilca,kp,br->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag 17
J21 += contract('ickl,jlcb,kp,ar->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2,o_delta,v_delta) #diag 17
J21 += -contract('cdal,jldb,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2,o_delta,v_delta) #diag 18
J21 += -contract('cdbl,ilda,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2,o_delta,v_delta) #diag 18
##-----------------------------------------------------------##
##--------------Higher order terms---------------------------##
J21 += contract('ickl,ka,jc,lp,br->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag 30
J21 += contract('jckl,kb,ic,lp,ar->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag 30
J21 += contract('ickl,ka,lb,jp,cr->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag 30
J21 += contract('jckl,kb,la,ip,cr->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag 30
J21 += contract('ickl,jc,lb,kp,ar->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag 30
J21 += contract('jckl,ic,la,kp,br->ijabpr',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,t1,o_delta,v_delta) #diag 30
J21 += -contract('dcbk,jd,ic,kp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,t1,o_delta,v_delta) #diag 31
J21 += -contract('dcak,id,jc,kp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,t1,o_delta,v_delta) #diag 31
J21 += -contract('dcbk,jd,ka,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,t1,o_delta,v_delta) #diag 31
J21 += -contract('dcak,id,kb,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,t1,o_delta,v_delta) #diag 31
J21 += -contract('dcbk,ic,ka,jp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,t1,o_delta,v_delta) #diag 31
J21 += -contract('dcak,jc,kb,ip,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,t1,o_delta,v_delta) #diag 31
J21 += -2*contract('dclk,ljdb,ic,kp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 32
J21 += -2*contract('dclk,lida,jc,kp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 32
J21 += -2*contract('dclk,ljdb,ka,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 32
J21 += -2*contract('dclk,lida,kb,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 32
J21 += contract('dclk,jldb,ic,kp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 33
J21 += contract('dclk,ilda,jc,kp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 33
J21 += contract('dclk,jldb,ka,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 33
J21 += contract('dclk,ilda,kb,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 33
J21 += contract('dclk,ijdb,lc,kp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 34
J21 += contract('dclk,jida,lc,kp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 34
J21 += contract('dclk,ijdb,ka,lp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 34
J21 += contract('dclk,jida,kb,lp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 34
J21 += contract('dclk,ljab,kd,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 35
J21 += contract('dclk,liba,kd,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 35
J21 += contract('dclk,ljab,ic,kp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 35
J21 += contract('dclk,liba,jc,kp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 35
J21 += contract('cdkl,kjad,ic,lp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 36
J21 += contract('cdkl,kibd,jc,lp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 36
J21 += contract('cdkl,kjad,lb,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 36
J21 += contract('cdkl,kibd,la,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 36
J21 += 0.5*contract('cdkl,klab,ic,jp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 37
J21 += 0.5*contract('cdkl,klba,jc,ip,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 37
J21 += 0.5*contract('cdkl,klab,jd,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 37
J21 += 0.5*contract('cdkl,klba,id,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 37
J21 += 0.5*contract('cdkl,ijcd,lb,kp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 38
J21 += 0.5*contract('cdkl,jicd,la,kp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 38
J21 += 0.5*contract('cdkl,ijcd,ka,lp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 38
J21 += 0.5*contract('cdkl,jicd,kb,lp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 38
J21 += contract('cdkl,kjdb,ic,lp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 39
J21 += contract('cdkl,kida,jc,lp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 39
J21 += contract('cdkl,kjdb,la,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 39
J21 += contract('cdkl,kida,lb,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 39
J21 += 0.5*contract('cdkl,ic,ka,jd,lp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,t1,o_delta,v_delta) #diag 40
J21 += 0.5*contract('cdkl,jc,kb,id,lp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,t1,o_delta,v_delta) #diag 40
J21 += 0.5*contract('cdkl,ic,ka,lb,jp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,t1,o_delta,v_delta) #diag 40
J21 += 0.5*contract('cdkl,jc,kb,la,ip,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,t1,o_delta,v_delta) #diag 40
J21 += 0.5*contract('cdkl,ic,jd,lb,kp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,t1,o_delta,v_delta) #diag 40
J21 += 0.5*contract('cdkl,jc,id,la,kp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,t1,o_delta,v_delta) #diag 40
J21 += 0.5*contract('cdkl,ka,jd,lb,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,t1,o_delta,v_delta) #diag 40
J21 += 0.5*contract('cdkl,kb,id,la,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,t1,o_delta,v_delta) #diag 40
J21 += -2*contract('dclk,kjab,ld,ip,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 38'
J21 += -2*contract('dclk,kiba,ld,jp,cr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 38'
J21 += -2*contract('dclk,kjab,ic,lp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 38'
J21 += -2*contract('dclk,kiba,jc,lp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta) #diag 38'
J21 += -2*contract('dclk,ijcb,ld,kp,ar->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 34'
J21 += -2*contract('dclk,jica,ld,kp,br->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 34'
J21 += -2*contract('dclk,ijcb,ka,lp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 34'
J21 += -2*contract('dclk,jica,kb,lp,dr->ijabpr',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,t1,o_delta,v_delta)  #diag 34'
##-----------------------------------------------------------##
'''
c =  np.reshape(J21,(occ*occ*virt*virt,occ*virt))
#print 'J21',J21.shape, c.shape


#differentiation w.r.t t2(p,q,r,s)
##------------------Linear terms--------------------------##
J22 = -np.einsum('ik,kp,jq,ar,bs->ijabpqrs',Fock_mo[:occ,:occ],o_delta,o_delta,v_delta,v_delta) #diag 1
J22 += -np.einsum('jk,kp,iq,br,as->ijabpqrs',Fock_mo[:occ,:occ],o_delta,o_delta,v_delta,v_delta) #diag 1
J22 += np.einsum('ca,ip,jq,cr,bs->ijabpqrs',Fock_mo[occ:nao,occ:nao],o_delta,o_delta,v_delta,v_delta) #diag 2
J22 += np.einsum('cb,jp,iq,cr,as->ijabpqrs',Fock_mo[occ:nao,occ:nao],o_delta,o_delta,v_delta,v_delta) #diag 2
J22 += 0.5*contract('cdab,ip,jq,cr,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],o_delta,o_delta,v_delta,v_delta) #diag 5
J22 += 0.5*contract('cdba,jp,iq,cr,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],o_delta,o_delta,v_delta,v_delta) #diag 5
J22 += 2*contract('jcbk,kp,iq,cr,as->ijabpqrs',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],o_delta,o_delta,v_delta,v_delta) #diag 6
J22 += 2*contract('icak,kp,jq,cr,bs->ijabpqrs',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],o_delta,o_delta,v_delta,v_delta) #diag 6
J22 += -contract('icka,kp,jq,cr,bs->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],o_delta,o_delta,v_delta,v_delta) #diag 7
J22 += -contract('jckb,kp,iq,cr,as->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],o_delta,o_delta,v_delta,v_delta) #diag 7
J22 += -contract('jcbk,ip,kq,cr,as->ijabpqrs',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],o_delta,o_delta,v_delta,v_delta)  #diag 8
J22 += -contract('icak,jp,kq,cr,bs->ijabpqrs',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],o_delta,o_delta,v_delta,v_delta)  #diag 8
J22 += 0.5*contract('ijkl,kp,lq,ar,bs->ijabpqrs',twoelecint_mo[:occ,:occ,:occ,:occ],o_delta,o_delta,v_delta,v_delta) #diag 9
J22 += 0.5*contract('jikl,kp,lq,br,as->ijabpqrs',twoelecint_mo[:occ,:occ,:occ,:occ],o_delta,o_delta,v_delta,v_delta) #diag 9
J22 += -contract('ickb,kp,jq,ar,cs->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],o_delta,o_delta,v_delta,v_delta) #diag 10
J22 += -contract('jcka,kp,iq,br,cs->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],o_delta,o_delta,v_delta,v_delta) #diag 10
##----------------------------------------------------------------------------------------------##
'''
##---------------------------------Non-linear terms---------------------------------------------##
J22 += -2*contract('idkl,ld,kp,jq,ar,bs->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 5
J22 += -2*contract('jdkl,ld,kp,iq,br,as->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 5
J22 += 2*contract('cdal,ld,ip,jq,cr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 6
J22 += 2*contract('cdbl,ld,jp,iq,cr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 6
J22 += -contract('cdak,kc,ip,jq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 7
J22 += -contract('cdbk,kc,jp,iq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 7
J22 += contract('idkl,kd,lp,jq,ar,bs->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 8
J22 += contract('jdkl,kd,lp,iq,br,as->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 8
J22 += contract('cjkl,ic,kp,lq,ar,bs->ijabpqrs',twoelecint_mo[occ:nao,:occ,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 9
J22 += contract('cikl,jc,kp,lq,br,as->ijabpqrs',twoelecint_mo[occ:nao,:occ,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 9
J22 += -contract('cdlb,ic,lp,jq,ar,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1,o_delta,o_delta,v_delta,v_delta) #diag 10
J22 += -contract('cdla,jc,lp,iq,br,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1,o_delta,o_delta,v_delta,v_delta) #diag 10
J22 += contract('cikl,kb,jp,lq,cr,as->ijabpqrs',twoelecint_mo[occ:nao,:occ,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 11
J22 += contract('cjkl,ka,ip,lq,cr,bs->ijabpqrs',twoelecint_mo[occ:nao,:occ,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 11
J22 += -contract('cdka,kb,jp,iq,cr,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1,o_delta,o_delta,v_delta,v_delta) #diag 12
J22 += -contract('cdkb,ka,ip,jq,cr,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1,o_delta,o_delta,v_delta,v_delta) #diag 12
J22 += contract('jclk,kb,lp,iq,cr,as->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 13
J22 += contract('iclk,ka,lp,jq,cr,bs->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 13
J22 += -contract('cdka,ic,kp,jq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1,o_delta,o_delta,v_delta,v_delta) #diag 14
J22 += -contract('cdkb,jc,kp,iq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1,o_delta,o_delta,v_delta,v_delta) #diag 14
J22 += -2*contract('jckl,kb,ip,lq,ar,cs->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 15
J22 += -2*contract('ickl,ka,jp,lq,br,cs->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 15
J22 += 2*contract('cdal,ic,lp,jq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 16
J22 += 2*contract('cdbl,jc,lp,iq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 16
J22 += contract('jckl,kb,ip,lq,cr,as->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 17
J22 += contract('ickl,ka,jp,lq,cr,bs->ijabpqrs',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 17
J22 += -contract('cdal,ic,jp,lq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 18 
J22 += -contract('cdbl,jc,ip,lq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1,o_delta,o_delta,v_delta,v_delta) #diag 18 
J22 += 2*contract('dclk,kica,jp,lq,br,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)   #diag 19 
J22 += 2*contract('dclk,kjcb,ip,lq,ar,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)   #diag 19 
J22 += 2*contract('dclk,jlbd,kp,iq,cr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)   #diag 19 
J22 += 2*contract('dclk,ilad,kp,jq,cr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)   #diag 19 
J22 += -contract('dckl,ljdb,kp,iq,cr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 20
J22 += -contract('dckl,lida,kp,jq,cr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 20
J22 += -contract('dckl,kica,lp,jq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 20
J22 += -contract('dckl,kjcb,lp,iq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 20
J22 += 0.5*contract('dclk,ikca,jp,lq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 21
J22 += 0.5*contract('dclk,jkcb,ip,lq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 21
J22 += 0.5*contract('dclk,jldb,ip,kq,cr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 21
J22 += 0.5*contract('dclk,ilda,jp,kq,cr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 21
J22 += contract('dckl,ildb,kp,jq,ar,cs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 22
J22 += contract('dckl,jlda,kp,iq,br,cs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 22
J22 += contract('dckl,kjac,ip,lq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 22 
J22 += contract('dckl,kibc,jp,lq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 22 
J22 += 0.5*contract('dckl,ijdc,kp,lq,ar,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 23 
J22 += 0.5*contract('dckl,jidc,kp,lq,br,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 23 
J22 += 0.5*contract('dckl,klab,ip,jq,dr,cs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 23 
J22 += 0.5*contract('dckl,klba,jp,iq,dr,cs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 23 
J22 += -2*contract('cdkl,klbd,jp,iq,cr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 24 
J22 += -2*contract('cdkl,klad,ip,jq,cr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 24 
J22 += -2*contract('cdkl,jica,kp,lq,br,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 24 
J22 += -2*contract('cdkl,ijcb,kp,lq,ar,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 24 
J22 += -2*contract('cdkl,ilcd,kp,jq,ar,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)  #diag 25 
J22 += -2*contract('cdkl,jlcd,kp,iq,br,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)  #diag 25 
J22 += -2*contract('cdkl,kjab,ip,lq,cr,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)  #diag 25 
J22 += -2*contract('cdkl,kiba,jp,lq,cr,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)  #diag 25 
J22 += contract('cdkl,klda,ip,jq,cr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 26 
J22 += contract('cdkl,kldb,jp,iq,cr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 26 
J22 += contract('cdkl,ijcb,kp,lq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 26 
J22 += contract('cdkl,jica,kp,lq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 26 
J22 += contract('dckl,lidc,kp,jq,ar,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)  #diag 27 
J22 += contract('dckl,ljdc,kp,iq,br,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)  #diag 27 
J22 += contract('dckl,kjab,lp,iq,dr,cs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)  #diag 27 
J22 += contract('dckl,kiba,lp,jq,dr,cs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta)  #diag 27 
J22 += -2*contract('dclk,kica,jp,lq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 28 
J22 += -2*contract('dclk,kjcb,ip,lq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 28 
J22 += -2*contract('dclk,jldb,kp,iq,cr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 28 
J22 += -2*contract('dclk,ilda,kp,jq,cr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 28 
J22 += contract('dckl,ikca,lp,jq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 29
J22 += contract('dckl,jkcb,lp,iq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 29
J22 += contract('dckl,ljdb,ip,kq,cr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 29
J22 += contract('dckl,lida,jp,kq,cr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2,o_delta,o_delta,v_delta,v_delta) #diag 29
##-------------------------------------------------------------------------------------------------------------------------##
##---------------------Higher order terms----------------------------------------------------------------------------------##
J22 += -2*contract('dclk,ic,ka,lp,jq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 32
J22 += -2*contract('dclk,jc,kb,lp,iq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 32
J22 += contract('dclk,ic,ka,jp,lq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 33
J22 += contract('dclk,jc,kb,ip,lq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 33
J22 += contract('dclk,lc,ka,ip,jq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta)  #diag 34
J22 += contract('dclk,lc,kb,jp,iq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta)  #diag 34
J22 += contract('dclk,ic,kd,lp,jq,ar,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 35
J22 += contract('dclk,jc,kd,lp,iq,br,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 35
J22 += contract('cdkl,ic,lb,kp,jq,ar,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta)  #diag 36
J22 += contract('cdkl,jc,la,kp,iq,br,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta)  #diag 36
J22 += 0.5*contract('cdkl,ic,jd,kp,lq,ar,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta)  #diag 37
J22 += 0.5*contract('cdkl,jc,id,kp,lq,br,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta)  #diag 37
J22 += 0.5*contract('cdkl,lb,ka,ip,jq,cr,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 38
J22 += 0.5*contract('cdkl,la,kb,jp,iq,cr,ds->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 38
J22 += contract('cdkl,ic,la,kp,jq,dr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 39
J22 += contract('cdkl,jc,lb,kp,iq,dr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 39
J22 += -2*contract('dclk,ld,ic,kp,jq,ar,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 38'
J22 += -2*contract('dclk,ld,jc,kp,iq,br,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta) #diag 38'
J22 += -2*contract('dclk,ld,ka,ip,jq,cr,bs->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta)  #diag 34'
J22 += -2*contract('dclk,ld,kb,jp,iq,cr,as->ijabpqrs',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1,o_delta,o_delta,v_delta,v_delta)  #diag 34'
##-------------------------------------------------------------------------------------------------------------------------##
'''
d = np.reshape(J22,(occ*occ*virt*virt,occ*occ*virt*virt))
#print 'J22',J22.shape ,d.shape


const1 = contract('ip,ra->iapr',o_delta,v_delta)
const2 = contract('ip,jq,ra,sb->ijabpqrs',o_delta,o_delta,v_delta,v_delta)

eta = 0.0
#J_11 = cp.deepcopy(const1)
#J_22 = cp.deepcopy(const2)


#J_11 = np.zeros((occ,virt,occ,virt))
J_11 = cp.deepcopy(J11) 
J_12 = np.zeros((occ,virt,red_occ,red_virt))
for i in range(0,occ):
  for a in range(0,virt):
    for p in range(0,occ):
      for q in range(0,p+1):
        for r in range(0,virt):
          for s in range(0,r+1):
            D1 = hf_mo_E[i]-hf_mo_E[a] + eta
            pq = q + (p*(p+1))/2
            rs = s + (r*(r+1))/2
            J_12[i,a,pq,rs] = J12[i,a,p,q,r,s]
J_11_new = np.reshape(J_11,(occ*virt,occ*virt))
J_12_new = np.reshape(J_12,(occ*virt,red_occ*red_virt))

J_21 = np.zeros((red_occ,red_virt,occ,virt))
J_22 = np.zeros((red_occ,red_virt,red_occ,red_virt))
for i in range(0,occ):
  for j in range(0,i+1):
    for a in range(0,virt):
      for b in range(0,a+1):
        for p in range(0,occ):
          for q in range(0,p+1):
            for r in range(0,virt):
              for s in range(0,r+1):
                D2 = hf_mo_E[i] + hf_mo_E[j] - hf_mo_E[a] - hf_mo_E[b] + 2*eta
                ij = j + (i*(i+1))/2
                ab = b + (a*(a+1))/2
                pq = q + (p*(p+1))/2
                rs = s + (r*(r+1))/2
                J_22[ij,ab,pq,rs] = J22[i,j,a,b,p,q,r,s]
J_22_new = np.reshape(J_22,(red_occ*red_virt,red_occ*red_virt))

for i in range(0,occ):
  for j in range(0,i+1):
    for a in range(0,virt):
      for b in range(0,a+1):
        for p in range(0,occ):
          for r in range(0,virt):
            ij = j + (i*(i+1))/2
            ab = b + (a*(a+1))/2
            J_21[ij,ab,p,r] = J21[i,j,a,b,p,r]
J_21_new = np.reshape(J_21,(red_occ*red_virt,occ*virt))

p = np.concatenate((J_11_new,J_12_new), axis=1)
q = np.concatenate((J_21_new,J_22_new), axis=1)
J = np.concatenate((p, q), axis=0)

w,v = np.linalg.eig(J) 
abs_w = abs(w)
min_w = np.amin(w)
print "Minimum eigen value",min_w

#round_off_w = np.around(abs_w, decimals = 2)
#unique_w = np.unique(round_off_w)
sort_w = sorted(abs_w)
#vector = v[:,4]
#print sort_w[4], vector

for i in range(len(w)):
  file1 = open("myfile.txt","a") #append mode
  file1.write('%s' %(i+1) +"\t"+'%s' %sort_w[i] + "\n")
  file1.close()

'''
exponent_max = np.log(sort_w[-1])
exponent_2_max = np.log(sort_w[-2])
exponent_3_max = np.log(sort_w[-3])
exponent_4_max = np.log(sort_w[-4])
exponent_5_max = np.log(sort_w[-5])
exponent_6_max = np.log(sort_w[-6])
exponent_7_max = np.log(sort_w[-7])
exponent_8_max = np.log(sort_w[-8])
exponent_9_max = np.log(sort_w[-9])
exponent_10_max = np.log(sort_w[-10])
exponent_11_max = np.log(sort_w[-11])
exponent_12_max = np.log(sort_w[-12])
exponent_13_max = np.log(sort_w[-13])
exponent_14_max = np.log(sort_w[-14])
exponent_15_max = np.log(sort_w[-15])
print exponent_max
print exponent_2_max
print exponent_3_max
print exponent_4_max
print exponent_5_max
print exponent_6_max
print exponent_7_max
print exponent_8_max
print exponent_9_max
print exponent_10_max
print exponent_11_max
print exponent_12_max
print exponent_13_max
print exponent_14_max
print exponent_15_max

'''
