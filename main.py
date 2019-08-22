# Import modules
import numpy as np
import copy as cp
import trans_mo
import MP2
import inp
import intermediates
import amplitude
import diis
import cc_symmetrize
import cc_update

mol = inp.mol
# Obtain the number of atomic orbitals in the basis set
nao = MP2.nao

# import important stuff
E_hf = trans_mo.E_hf
Fock_mo = MP2.Fock_mo
twoelecint_mo = MP2.twoelecint_mo 
t1 = MP2.t1 
D1 = MP2.D1
t2 = MP2.t2
D2=MP2.D2
So = MP2.So
Do=MP2.Do
Sv = MP2.Sv
Dv=MP2.Dv
occ = MP2.occ
virt = MP2.virt
E_old = MP2.E_mp2_tot
n_iter = inp.n_iter
calc = inp.calc
conv = 10**(-inp.conv)
max_diis = inp.max_diis
#nfo = MP2.nfo
#nfv = MP2.nfv

#    Evaluate the energy
def energy_ccd(t2):
  E_ccd = 2*np.einsum('ijab,ijab',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) - np.einsum('ijab,ijba',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
  return E_ccd

def energy_ccsd(t1,t2):
  E_ccd = energy_ccd(t2)
  E_ccd += 2*np.einsum('ijab,ia,jb',twoelecint_mo[:occ,:occ,occ:nao,occ:nao],t1,t1) - np.einsum('ijab,ib,ja',twoelecint_mo[:occ,:occ,occ:nao,occ:nao],t1,t1)
  return E_ccd

def convergence_I(E_ccd,E_old,eps_t,eps_So,eps_Sv):
  del_E = E_ccd - E_old
  if abs(eps_t) <= conv and abs(eps_So) <= conv and abs(eps_Sv) <= conv and abs(del_E) <= conv:
    print "ccd converged!!!"
    print "Total energy is : "+str(E_hf + E_ccd)
    return True
  else:
    print "cycle number : "+str(x+1)
    print "change in t1+t2 , So, Sv : "+str(eps_t)+" "+str(eps_So)+" "+str(eps_Sv)
    print "energy difference : "+str(del_E)
    print "energy : "+str(E_hf + E_ccd)
    E_old = E_ccd
    return False

def convergence(E_ccd,E_old,eps):
  del_E = E_ccd - E_old
  if abs(eps) <= conv and abs(del_E) <= conv:
    print "ccd converged!!!"
    print "Total energy is : "+str(E_hf + E_ccd)
    return True
  else:
    print "cycle number : "+str(x+1)
    print "change in t1 and t2 : "+str(eps)
    print "energy difference : "+str(del_E)
    print "energy : "+str(E_hf + E_ccd)
    return False

### Setup DIIS
if inp.diis == True:
  diis_vals_t2, diis_errors_t2 = diis.DIIS_ini(t2)
  diis_errors = []
  if calc == 'CCSD' or calc == 'ICCSD' or calc == 'ICCSD-PT':
    diis_vals_t1, diis_errors_t1 = diis.DIIS_ini(t1)
  if calc == 'ICCD' or calc == 'ILCCD' or calc == 'ICCSD':
    diis_vals_So, diis_errors_So = diis.DIIS_ini(So)
    diis_vals_Sv, diis_errors_Sv = diis.DIIS_ini(Sv)

for x in range(0,n_iter):
  #t2 = np.zeros((occ,occ,virt,virt))
  if calc == 'LCCD':
    print "-----------LCCD-------------"
    I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = intermediates.initialize()
    R_ijab = amplitude.doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,t2,t2)
    R_ijab = amplitude.symmetrize(R_ijab)
    oldt2 = t2.copy()
    eps_t, t2 = amplitude.update_t2(R_ijab,t2)
    if inp.diis == True:
      if x+1>max_diis:
        # Limit size of DIIS vector
        if (len(diis_vals_t2) > max_diis):
          del diis_vals_t2[0]
          del diis_errors[0]
        diis_size = len(diis_vals_t2) - 1

        # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        ci = diis.error_matrix(diis_size,diis_errors)
    
        # Calculate new amplitudes
        if (x+1) % max_diis == 0:
          t2 = diis.new_amp(t2,diis_size,ci,diis_vals_t2)
        # End DIIS amplitude update
    E_ccd = energy_ccd(t2)
    val = convergence(E_ccd,E_old,eps_t)
    if val == True :
      break
    else:  
      E_old = E_ccd
    if inp.diis == True: 
      # Add DIIS vectors
      error_t2, diis_vals_t2 = diis.errors(t2,oldt2,diis_vals_t2)
      # Build new error vector
      diis_errors.append((error_t2))
  if calc == 'CCD':
    print "-----------CCD------------"
    I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = intermediates.initialize()
    I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = intermediates.update_int(t2,t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)
    R_ijab = amplitude.doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,t2,t2)
    R_ijab = amplitude.symmetrize(R_ijab)
    oldt2 = t2.copy()
    eps_t, t2 = amplitude.update_t2(R_ijab,t2)
    if inp.diis == True:
      if x+1>max_diis:
        # Limit size of DIIS vector
        if (len(diis_vals_t2) > max_diis):
          del diis_vals_t2[0]
          del diis_errors[0]
        diis_size = len(diis_vals_t2) - 1

        # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        ci = diis.error_matrix(diis_size,diis_errors)
    
        # Calculate new amplitudes
        if (x+1) % max_diis == 0:
          t2 = diis.new_amp(t2,diis_size,ci,diis_vals_t2)
        # End DIIS amplitude update
    E_ccd = energy_ccd(t2)
    val = convergence(E_ccd,E_old,eps_t)
    if val == True :
      break
    else:  
      E_old = E_ccd
    if inp.diis == True: 
      # Add DIIS vectors
      error_t2, diis_vals_t2 = diis.errors(t2,oldt2,diis_vals_t2)
      # Build new error vector
      diis_errors.append((error_t2))

  if calc == 'CCSD':
    print "-----------CCSD------------"
    tau = cp.deepcopy(t2)
    tau += np.einsum('ia,jb->ijab',t1,t1)
    
    I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = intermediates.initialize()
    I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = intermediates.update_int(tau,t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)
    I1, I2 = intermediates.R_ia_intermediates(t1)
 
    R_ia = amplitude.singles(I1,I2,I_oo,I_vv,tau,t1,t2)
    I_oo,I_vv,I_oovo,I_vovv,Ioooo_2,I_voov,Iovov_3,Iovvo_3,Iooov,I3=intermediates.singles_intermediates(t1,t2,I_oo,I_vv,I2)
    II_a, II_b, II_c, II_d = intermediates.disconnected_t2_terms(Ivvvv,t1)
    R_ijab = amplitude.doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,t2)
    R_ijab += amplitude.singles_n_doubles(t1,I_oovo,I_vovv,II_a,II_b,II_c,II_d)
    #R_ijab += amplitude.higher_order(t1,t2,Iooov,I3,Ioooo_2,I_voov,II_d,II_e)
    #R_ijab += amplitude.higher_order(t1,t2,II_d,II_e)
    R_ijab = cc_symmetrize.symmetrize(R_ijab)
    
    oldt2 = t2.copy()
    oldt1 = t1.copy()
    eps_t, t1, t2 = cc_update.update_t1t2(R_ia,R_ijab,t1,t2)
    
    if inp.diis == True:
      if x+1>max_diis:
        # Limit size of DIIS vector
        if (len(diis_vals_t1) > max_diis):
          del diis_vals_t1[0]
          del diis_vals_t2[0]
          del diis_errors[0]
        diis_size = len(diis_vals_t1) - 1

        # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        ci = diis.error_matrix(diis_size,diis_errors)
    
        # Calculate new amplitudes
        if (x+1) % max_diis == 0:
          t1 = diis.new_amp(t1,diis_size,ci,diis_vals_t1)
          t2 = diis.new_amp(t2,diis_size,ci,diis_vals_t2)
        # End DIIS amplitude update
    E_ccd = energy_ccsd(t1,t2)
    val = convergence(E_ccd,E_old,eps_t)
    if val == True :
      break
    else:  
      E_old = E_ccd
    if inp.diis == True: 
      # Add DIIS vectors
      error_t1, diis_vals_t1 = diis.errors(t1,oldt1,diis_vals_t1)
      error_t2, diis_vals_t2 = diis.errors(t2,oldt2,diis_vals_t2)
      # Build new error vector
      diis_errors.append(np.concatenate((error_t1,error_t2)))

  if calc == 'ICCSD':
    print "----------ICCSD------------"
    tau = cp.deepcopy(t2)
    #tau += np.einsum('ia,jb->ijab',t1,t1)
    I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2, I_oovo, I_vovv = intermediates.initialize()

    #I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = intermediates.update_int(tau,t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)

    I1, I2 = intermediates.R_ia_intermediates(t1)
    II_oo = intermediates.W1_int_So(So)
    II_vv = intermediates.W1_int_Sv(Sv)
    
    R_ia = amplitude.singles(I1,I2,I_oo,I_vv,tau,t1,t2)
    #I_oo,I_vv,I_oovo,I_vovv,Ioooo_2,I_voov,Iovov_3,Iovvo_3,Iooov,I3=intermediates.singles_intermediates(t1,t2,tau,I_oo,I_vv,I2)
    R_ijab = amplitude.doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,t2)
    #R_ijab += amplitude.singles_n_doubles(t1,t2,tau,I_oovo,I_vovv,Iovov_3,Iovvo_3,Iooov,I3,Ioooo_2,I_voov)
    R_ijab += amplitude.singles_n_doubles(t1,t2,tau,I_oovo,I_vovv) #needs to be removed
    R_ijab += amplitude.inserted_diag_So(t2,II_oo) 
    R_ijab += amplitude.inserted_diag_Sv(t2,II_vv) 
    R_ijab = cc_symmetrize.symmetrize(R_ijab)
    
    R_iuab = amplitude.Sv_diagram_vs_contraction(Sv,II_vv)
    R_iuab += amplitude.Sv_diagram_vt_contraction(t2)
    R_ijav = amplitude.So_diagram_vs_contraction(So,II_oo)
    R_ijav += amplitude.So_diagram_vt_contraction(t2)
    
    oldt2 = t2.copy()
    oldt1 = t1.copy()
    oldSo = So.copy()
    oldSv = Sv.copy()
    eps_t, t1, t2 = cc_update.update_t1t2(R_ia,R_ijab,t1,t2)
    eps_So, So = cc_update.update_So(R_ijav,So)
    eps_Sv, Sv = cc_update.update_Sv(R_iuab,Sv)
    if inp.diis == True:
      if x+1>max_diis:
        # Limit size of DIIS vector
        if (len(diis_vals_t1) > max_diis):
          del diis_vals_t1[0]
          del diis_vals_t2[0]
          del diis_errors[0]
          del diis_vals_So[0]
          del diis_errors_So[0]
          del diis_vals_Sv[0]
          del diis_errors_Sv[0]
        diis_size = len(diis_vals_t1) - 1
        diis_size_So = len(diis_vals_So) - 1
        diis_size_Sv = len(diis_vals_Sv) - 1

        # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        ci = diis.error_matrix(diis_size,diis_errors)
        ci_So = diis.error_matrix(diis_size_So,diis_errors_So)
        ci_Sv = diis.error_matrix(diis_size_Sv,diis_errors_Sv)
    
        # Calculate new amplitudes
        if (x+1) % max_diis == 0:
          t1 = diis.new_amp(t1,diis_size,ci,diis_vals_t1)
          t2 = diis.new_amp(t2,diis_size,ci,diis_vals_t2)
          So = diis.new_amp(So,diis_size_So,ci_So,diis_vals_So)
          Sv = diis.new_amp(Sv,diis_size_Sv,ci_Sv,diis_vals_Sv)
        # End DIIS amplitude update
    E_ccd = energy_ccsd(t1,t2)
    val = convergence_I(E_ccd,E_old,eps_t,eps_So,eps_Sv)
    if val == True :
      break
    else:  
      E_old = E_ccd
    if inp.diis == True: 
      # Add DIIS vectors
      error_t1, diis_vals_t1 = diis.errors(t1,oldt1,diis_vals_t1)
      error_t2, diis_vals_t2 = diis.errors(t2,oldt2,diis_vals_t2)
      error_So, diis_vals_So = diis.errors(So,oldSo,diis_vals_So)
      error_Sv, diis_vals_Sv = diis.errors(Sv,oldSv,diis_vals_Sv)
      # Build new error vector
      diis_errors.append(np.concatenate((error_t1,error_t2)))
      diis_errors_So.append((error_So))
      diis_errors_Sv.append((error_Sv))

  if calc == 'ICCSD-PT':
    print "----------ICCSD-PT------------"
    tau = cp.deepcopy(t2)
    tau += np.einsum('ia,jb->ijab',t1,t1)
    I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = intermediates.initialize()

    I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = intermediates.update_int(tau,t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)

    I1, I2 = intermediates.R_ia_intermediates(t1)
    II_oo, II_vv = intermediates.W1_intermediates(So,Sv)
    R_ia = amplitude.singles(I1,I2,I_oo,I_vv,tau,t1,t2)
    I_oo,I_vv,I_oovo,I_vovv,Ioooo_2,I_voov,Iovov_3,Iovvo_3,Iooov,I3=intermediates.singles_intermediates(t1,t2,tau,I_oo,I_vv,I2)
    R_ijab = amplitude.doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,t2)
    R_ijab += amplitude.singles_n_doubles(t1,t2,tau,I_oovo,I_vovv,Iovov_3,Iovvo_3,Iooov,I3,Ioooo_2,I_voov)
    R_ijab += amplitude.S_diagrams(t2,II_oo,II_vv) 
    R_ijab = amplitude.symmetrize(R_ijab)
    
    oldt2 = t2.copy()
    oldt1 = t1.copy()
    eps_t, t1, t2 = amplitude.update_t1t2(R_ia,R_ijab,t1,t2)
    if inp.diis == True:
      if x+1>max_diis:
        # Limit size of DIIS vector
        if (len(diis_vals_t1) > max_diis):
          del diis_vals_t1[0]
          del diis_vals_t2[0]
          del diis_errors[0]
        diis_size = len(diis_vals_t1) - 1

        # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        ci = diis.error_matrix(diis_size,diis_errors)
    
        # Calculate new amplitudes
        if (x+1) % max_diis == 0:
          t1 = diis.new_amp(t1,diis_size,ci,diis_vals_t1)
          t2 = diis.new_amp(t2,diis_size,ci,diis_vals_t2)
        # End DIIS amplitude update
    E_ccd = energy_ccsd(t1,t2)
    val = convergence(E_ccd,E_old,eps_t)
    if val == True :
      break
    else:  
      E_old = E_ccd
    if inp.diis == True: 
      # Add DIIS vectors
      error_t1, diis_vals_t1 = diis.errors(t1,oldt1,diis_vals_t1)
      error_t2, diis_vals_t2 = diis.errors(t2,oldt2,diis_vals_t2)
      # Build new error vector
      diis_errors.append(np.concatenate((error_t1,error_t2)))
