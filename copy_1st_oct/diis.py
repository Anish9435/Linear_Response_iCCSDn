import numpy as np
from numpy import linalg


### Setup DIIS
def DIIS_ini(A):
  diis_vals_A = [A.copy()]
  diis_errors = []
  return diis_vals_A, diis_errors

def errors(A,oldA,diis_vals_A):
  diis_vals_A.append(A.copy())
  # Build new error vector
  error_A = (A - oldA).ravel()
  return error_A,diis_vals_A

# Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
def error_matrix(diis_size,diis_errors):
  B = np.ones((diis_size + 1, diis_size + 1)) * -1
  B[-1, -1] = 0
  for n1, e1 in enumerate(diis_errors):
    for n2, e2 in enumerate(diis_errors):
      # Vectordot the error vectors
      B[n1, n2] = np.dot(e1, e2)
  B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
  # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
  resid = np.zeros(diis_size + 1)
  resid[-1] = -1
  print resid
  # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
  ci = np.linalg.solve(B, resid)
  return ci

def new_amp(A,diis_size,ci,diis_vals_A): 
  A[:] = 0
  for num in range(diis_size):
    A += ci[num] * diis_vals_A[num + 1]
  return A

