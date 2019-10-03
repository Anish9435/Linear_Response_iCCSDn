import trans_mo
from pyscf import cc
from pyscf.cc import ccsd_t

mf = trans_mo.mf
mycc = cc.CCSD(mf).set(max_cycle=200)
mycc.kernel()
t2 = mycc.t2
#print mycc.kernel()[0]
print ccsd_t.kernel(mycc,mycc.ao2mo())
