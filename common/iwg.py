import numpy as np
import scipy.linalg as sl
from scipy.special import gamma, digamma, gammaln

def gaussian_gradx_loglikelihood(x, th, Sig):
  x = np.atleast_2d(x)
  th = np.atleast_2d(th)
  Siginv = np.linalg.inv(Sig)
  return (th.dot(Siginv) - x.dot(Siginv))

def gaussian_for_IW_loglikelihood(x, Sig, mu):
  x = np.atleast_2d(x)
  Sig = np.atleast_2d(Sig)
  return -.5*(-x.shape[1]/np.log(2.*np.pi) - .5*np.linalg.slogdet(Sig)[1] - .5*np.einsum('ij,kjr,ir->ik', x-mu, np.linalg.inv(Sig), x-mu))

def KL_IW(v1, V1, v2, V2):
  d = V1.shape[0]
  sign1, logdetV1 = np.linalg.slogdet(V1)
  sign2, logdetV2 = np.linalg.slogdet(V2)
  t1 = .5*v2*(logdetV1-logdetV2)
  js = np.arange(1, d+1)
  t2 = gammaln( 0.5*(v2+1-js) ).sum() - gammaln( 0.5*(v1+1-js) ).sum()
  t3 = .5*(v1-v2)*digamma( 0.5*(v1+1-js) ).sum()
  t4 = -.5*d*v1 + .5*v1*np.trace(np.linalg.solve(V1, V2))
  return t1+t2+t3+t4




