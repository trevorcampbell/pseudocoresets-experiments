import numpy as np
import pickle as pk
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..')) # read library from local folder: can be removed if it's install systemwide
import bayesiancoresets as bc
from scipy.stats import multivariate_normal
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import model_linreg

def build_synthetic_dataset(N, w, noise_std=0.1):
  d = len(w)
  x = np.random.randn(N, d)
  x[:,-1]=1.
  y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
  return x, y

nm = sys.argv[1]
tr = sys.argv[2]
results_fldr = 'results'
if not os.path.exists(results_fldr):
  os.mkdir(results_fldr)

#use the trial # as seed
np.random.seed(int(tr))
M = 300
SVI_opt_itrs = 1000
BPSVI_opt_itrs = 1000
n_subsample_opt = 200
n_subsample_select = 10000
proj_dim = 100
pihat_noise = 0.75
BPSVI_step_sched = lambda m: lambda i : 0.1/(1+i)
SVI_step_sched = lambda i : 1./(1+i)

N = 2000  # number of data points
D = 100  # number of features
d = D+1 # dimensionality of w

w_true = np.random.randn(d)
X, Y = build_synthetic_dataset(N, w_true)
Z = np.hstack((X, Y[:,np.newaxis]))

#get empirical mean/std
datastd = Y.std()
datamn = Y.mean()

#model params
mu0 = datamn*np.ones(d)
ey = np.eye(d)
Sig0 = (datastd**2+datamn**2)*ey
Sig0inv = np.linalg.inv(Sig0)
#get true posterior
mup, LSigp, LSigpInv = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, Z, np.ones(X.shape[0]))
Sigp = LSigp.dot(LSigp.T)
SigpInv = LSigpInv.T.dot(LSigpInv)

#create function to output log_likelihood given param samples
print('Creating log-likelihood function')
log_likelihood = lambda z, th : model_linreg.gaussian_loglikelihood(z, th, datastd**2)

print('Creating gradient log-likelihood function')
grad_log_likelihood = lambda z, th : model_linreg.gaussian_grad_x_loglikelihood(z, th, datastd**2)

#create tangent space for well-tuned Hilbert coreset alg
print('Creating tuned projector for Hilbert coreset construction')
sampler_optimal = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(LSigp.T)
prj_optimal = bc.BlackBoxProjector(sampler_optimal, proj_dim, log_likelihood, grad_log_likelihood)

#create tangent space for poorly-tuned Hilbert coreset alg
print('Creating untuned projector for Hilbert coreset construction')
U = np.random.rand()
muhat = U*mup + (1.-U)*mu0
Sighat = U*Sigp + (1.-U)*Sig0
#now corrupt the smoothed pihat
muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
Sighat *= np.exp(-2*pihat_noise*np.fabs(np.random.randn()))
LSighat = np.linalg.cholesky(Sighat)

sampler_realistic = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(LSighat.T)
prj_realistic = bc.BlackBoxProjector(sampler_realistic, proj_dim, log_likelihood, grad_log_likelihood)

print('Creating black box projector')
def sampler_w(n, wts, pts):
    if pts.shape[0] == 0:
      wts = np.zeros(1)
      pts = np.zeros((1, Z.shape[1]))
    muw, LSigw, LSigwInv = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, pts, wts)
    return muw + np.random.randn(n, muw.shape[0]).dot(LSigw.T)
prj_w = bc.BlackBoxProjector(sampler_w, proj_dim, log_likelihood, grad_log_likelihood)

#create coreset construction objects
print('Creating coreset construction objects')
sparsevi = bc.SparseVICoreset(Z, prj_w, opt_itrs = SVI_opt_itrs, n_subsample_opt = n_subsample_opt,  n_subsample_select = n_subsample_select, step_sched = SVI_step_sched)
giga_optimal = bc.HilbertCoreset(Z, prj_optimal)
giga_realistic = bc.HilbertCoreset(Z, prj_realistic)
unif = bc.UniformSamplingCoreset(Z)

algs = {'BPSVI': None,
        'SVI': sparsevi,
        'GIGAO': giga_optimal,
        'GIGAR': giga_realistic,
        'RAND': unif,
        'PRIOR': None}
alg = algs[nm]

print('Building coreset')
#build coresets
w = [np.array([0.])]
p = [np.zeros((1, Z.shape[1]))]

def build_per_m(m): # construction in parallel for different coreset sizes used in BPSVI
  print('building for m=', m)
  bpsvi = bc.BatchPSVICoreset(Z, prj_w, opt_itrs = BPSVI_opt_itrs, n_subsample_opt = n_subsample_opt, step_sched = BPSVI_step_sched(m))
  bpsvi.build(m)
  print('built for m=',m)
  return bpsvi.get()

if nm in ['BPSVI']:
  from multiprocessing import Pool
  pool = Pool(processes=4)
  res = pool.map(build_per_m, range(1, M+1))
  i=1
  for (wts, pts, _) in res:
    w.append(wts)
    p.append(pts)
    i+=1
else:
  for m in range(1, M+1):
    if nm!='PRIOR':
      print('trial: ' + str(tr) +' alg: ' + nm + ' ' + str(m) +'/'+str(M))
      alg.build(1)
      #store weights
      wts, pts, idcs = alg.get()
      w.append(wts)
      p.append(pts)
    else:
      w.append(np.array([0.]))
      p.append(np.zeros((1,Y.shape[0])))

muw = np.zeros((M+1, mu0.shape[0]))
Sigw = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
rklw = np.zeros(M+1)
fklw = np.zeros(M+1)
if nm!='PRIOR':
  for m in range(M+1):
    muw[m, :], LSigw, LSigwInv = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, p[m], w[m])
    Sigw[m, :, :] = LSigw.dot(LSigw.T)
    rklw[m] = model_linreg.gaussian_KL(muw[m,:], Sigw[m,:,:], mup, SigpInv)
    fklw[m] = model_linreg.gaussian_KL(mup, Sigp, muw[m,:], LSigwInv.T.dot(LSigwInv))
else:
  for m in range(M+1):
    muw[m, :], Sigw[m,:,:] = mu0, Sig0
    rklw[m] = model_linreg.gaussian_KL(muw[m,:], Sigw[m,:,:], mup, SigpInv)
    fklw[m] = model_linreg.gaussian_KL(mup, Sigp, muw[m,:], Sigw[m,:,:])

f = open(results_fldr+'/results_'+str(nm)+'_'+str(tr)+'.pk', 'wb')
res = (w, rklw, fklw)
pk.dump(res, f)
f.close()
