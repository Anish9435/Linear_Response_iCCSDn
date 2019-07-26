import numpy as np
import copy as cp
import main
import MP2
import davidson
import inp

t1 = main.t1
t2 = main.t2
occ = MP2.occ
virt = MP2.virt
o_act = inp.o_act
v_act = inp.v_act
t1_new,t2_new,So_new,Sv_new = davidson.guess_X(occ,virt,o_act,v_act)

factor_t2 = (np.eye((occ*occ*virt*virt),dtype=float)-np.outer(t2,t2))
t2_proj_out = np.dot(factor_t2,np.reshape(t2_new,((occ*occ*virt*virt),1)))
t2_new = cp.deepcopy(np.reshape(t2_proj_out,(occ,occ,virt,virt))) 
print np.shape(t2_new)
print np.shape(t2_proj_out)
#print np.shape(p)
#print np.shape(q)



y = np.reshape(t2,(100,1))
z = np.reshape(y,(5,5,2,2))
#print t2
#print y
#print z
