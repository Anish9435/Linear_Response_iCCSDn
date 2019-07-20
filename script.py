from subprocess import call
from pyscf import gto, scf, ao2mo, cc
from pyscf.cc import ccsd_t

call(["rm","inp-*"])
inp_part1 = """#Import module
import pyscf.gto

# Specify geometry and basis set
mol = pyscf.gto.M(
verbose = 5,
atom ='''"""

def inp_part3(basis,calc,conv,n_iter,diis,max_diis,o_act,v_act,nfo,nfv):
  content ="""''',                        
basis = '"""+basis+"""',
symmetry = True)

# Specify linear or non-linear
calc = '"""+calc+"""'
# Specify convergence criteria
conv = """+str(conv)+"""
# Specify number of iteration
n_iter = """+str(n_iter)+"""
#DIIS or not
diis = """+str(diis)+"""
max_diis = """+str(max_diis)+"""

# Specify number of active orbitals
o_act = """+str(o_act)+"""
v_act = """+str(v_act)+"""

# specify number of frozen orbitals
nfo = """+str(nfo)+"""
nfv = """+str(nfv)
  return content


geom_file = open("geom.xyz","r")
lines = geom_file.readlines()
natom = len(lines)
files = []

##### Creating input file for the cluster #####
f_name = "inp-cluster.py"
files.append(f_name)
inp_file = open(f_name,"a")
inp_file.write(inp_part1)

for i in range(2,natom):
  inp_file.write(lines[i]) 
inp_file.write(inp_part3('cc-pVDZ','ICCSD',8,200,True,7,4,4,4,5))   # <----Write specifications of calculations here for cluster
inp_file.close()

atom_num_monomer = 6             # <----- specify the number of atoms in the monomer unit. E.g. 3 for water

num_monomer = natom/atom_num_monomer

print "This cluster has "+str(num_monomer)+" monomers"

##### Creating input file for monomers #####
for i in range(0,num_monomer):
  f_name = "inp-monomer-"+str(i+1)+".py"
  files.append(f_name)
  f = open(f_name,"a")
  f.write(inp_part1)
  for j in range(i*atom_num_monomer,(i+1)*atom_num_monomer):
    f.write(lines[2:][j])
  f.write(inp_part3('cc-pVDZ','ICCSD',8,200,True,7,2,2,4,5))                      # <----Write specifications of calculations here for monomer
  f.close()

res = []
res.append("cc-pVDZ, Cluster(4,4), Monomer(2,2)")       # <----Write specifications of calculations here
res.append('\n')
for i in range(len(files)):
  res.append("            "+files[i]) 
res.append('\n')
res.append("iCCSDn    ")                              # <---- The type of calculation you are running

E_mono=0
for x in files[:]:
  call(["cp",x,"inp.py"])
  y = x[:-3]+"-result.dat"
  print y
  with open(y,"w+") as output:
    call(["python", "./main.py"], stdout=output)   # <-- code name
  f = open(y,"r")
  lines = f.readlines()
  for line in reversed(lines):
    if "Total energy is : " in line:
      (i,i,i,i,val) = line.split()
      if files.index(x) == 0:
        E_clus = float(val)
      else:
        E_mono += float(val)
      res.append(val)
      res.append("         ")
      break
  f.close()

E_st = (E_clus-E_mono)*627.5095
res.append('\n')
res.append("Stabilization energy   ")
res.append(str(E_st))
res.append(" kcal/mol")
res.append('\n')

######   CCSD  ######
res.append("CCSD      ")
E_mono = 0
for x in files[:]:
  call(["cp",x,"inp.py"])
  y = x[:-3]+"-ccsd-result.dat"
  with open(y,"w+") as output:
    call(["python", "./ccsd_t.py"], stdout=output)   # <-- code name
  f = open(y,"r")
  lines = f.readlines()
  for line in reversed(lines):
    if "E(CCSD) =" in line:
      (i,i,val,i,i,i) = line.split()
      if files.index(x) == 0:
        E_clus = float(val)
      else:
        E_mono += float(val)
      res.append(val)
      res.append("         ")
      break
  f.close()

E_st = (E_clus-E_mono)*627.5095
res.append('\n')
res.append("Stabilization energy   ")
res.append(str(E_st))
res.append(" kcal/mol")
res.append('\n')

       
######   CCSD(T)  ######
res.append("CCSD(T)   ")
E_mono = 0
for x in files[:]:
  call(["cp",x,"inp.py"])
  y = x[:-3]+"-ccsd-result.dat"
  f = open(y,"r")
  lines = f.readlines()
  for line in reversed(lines):
    if "CCSD(T) correction =" in line:
      (i,i,i,val) = line.split()
      E_ccsd_t = float(val)
    if "E(CCSD) =" in line:
      (i,i,val,i,i,i) = line.split()
      E_ccsd = float(val)
      if files.index(x) == 0:
        E_clus = E_ccsd_t + E_ccsd
      else:
        E_mono += E_ccsd_t + E_ccsd
      res.append(str(E_ccsd_t+E_ccsd))
      res.append("         ")
      break
  f.close()

E_st = (E_clus-E_mono)*627.5095
res.append('\n')
res.append("Stabilization energy   ")
res.append(str(E_st))
res.append(" kcal/mol")
res.append('\n')
res.append('\n')

#call(["rm",y])

f = open("result.dat","a")                           ## <--- change name of result file here
for line in res:
  out=line
  f.write(out)
f.close()
del res[:]
