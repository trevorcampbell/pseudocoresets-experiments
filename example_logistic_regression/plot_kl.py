import bokeh.plotting as bkp
from bokeh.io import export_png
import bokeh.layouts as bkl
import pickle as pk
import numpy as np
import sys, os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *

dnm=sys.argv[1]
fldr_figs=sys.argv[2]
fldr_res=sys.argv[3]

algs = [('BPSVI', 'PSVI', pal[-1]), ('DPBPSVI', 'DP-PSVI', pal[-2]), ('SVI', 'SparseVI', pal[0]), ('RAND', 'Uniform', pal[3]), ('GIGAO','GIGA (Optimal)', pal[1]), ('GIGAR','GIGA (Realistic)', pal[2])]
fldr_figs = 'figs'
if not os.path.exists(fldr_figs):
  os.mkdir(fldr_figs)

figs=[]
print('Plotting ' + dnm)
fig = bkp.figure(y_axis_type='log', y_axis_label='Reverse KL',  x_axis_label='# Iterations', plot_height=1500, plot_width=2000, toolbar_location=None)
preprocess_plot(fig, '72pt', False, True)
fig2 = bkp.figure(y_axis_type='log', y_axis_label='', x_axis_type='log', x_axis_label='CPU Time (s)', plot_height=1500, plot_width=2000, toolbar_location=None)
preprocess_plot(fig2, '72pt', True, True)
fig3 = bkp.figure(y_axis_type='log', y_axis_label='',  x_axis_label='Coreset Size', plot_height=1500, plot_width=2000, toolbar_location=None)
preprocess_plot(fig3, '72pt', False, True)
figs.append([fig, fig2, fig3])

#get normalizations based on the prior
std_kls = {}
exp_prfx = ''
trials = [fn for fn in os.listdir(fldr_res) if fn.startswith(dnm+'_PRIOR')]
if len(trials) == 0:
  print('Need to run prior to establish baseline first')
  quit()
kltot = 0.
M = 0
for tridx, fn in enumerate(trials):
  f = open(os.path.join(fldr_res,fn), 'rb')
  res = pk.load(f) #(cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace)
  f.close()
  assert np.all(res[5] == res[5][0]) #make sure prior doesn't change...
  kltot += res[5][0]
  M = res[0].shape[0]
std_kls[dnm] = kltot / len(trials)
kl0=std_kls[dnm]

for alg in algs:
  trials = [fn for fn in os.listdir(fldr_res) if fn.startswith(dnm+'_'+alg[0])]
  if len(trials) == 0:
    fig.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    fig2.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig2.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    fig3.line([], [], color=alg[2], legend_label=alg[1], line_width=10); fig3.patch([], [], color=alg[2], legend_label=alg[1], alpha=0.3)
    continue
  kls = np.zeros((len(trials), M))
  cputs = np.zeros((len(trials), M))
  cszs = np.zeros((len(trials), M))
  for tridx, fn in enumerate(trials):
    f = open(os.path.join(fldr_res, fn), 'rb')
    res = pk.load(f) #(cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace)
    f.close()
    cputs[tridx, :] = res[0]
    wts = res[1]
    pts = res[2]
    mu = res[3]
    Sig = res[4]
    kl = res[5]
    cszs[tridx, :] = np.array([len(w) for w in wts])
    kls[tridx, :] = kl/kl0
    if max(kls[tridx, :])>10: # shooting off to 10x prior KLD: sometimes can occur with GIGAR
      kls[tridx, :]= None
    if 'PRIOR' in fn:
      kls[tridx, :] = np.median(kls[tridx,:])

  #cputs[0] is setup time
  #cputs[i] is time for iteration i in 1:M
  # for BPSVI, since each iter runs independently, need to add cputs[0] to all scputs[i]
  # for all others, since iters build up on each other, need to cumulative sum
  if alg[0] in ['BPSVI', 'DPBPSVI']:
    cputs[:, 1:] += cputs[:, 0][:,np.newaxis]
  else:
    cputs = np.cumsum(cputs, axis=1)

  cput50 = np.percentile(cputs, 50, axis=0)
  cput25 = np.percentile(cputs, 25, axis=0)
  cput75 = np.percentile(cputs, 75, axis=0)

  csz50 = np.percentile(cszs, 50, axis=0)
  csz25 = np.percentile(cszs, 25, axis=0)
  csz75 = np.percentile(cszs, 75, axis=0)

  kl50 = np.percentile(kls, 50, axis=0)
  kl25 = np.percentile(kls, 25, axis=0)
  kl75 = np.percentile(kls, 75, axis=0)

  if alg[0]=='BPSVI':
      kl50bpsvi = np.copy(kl50)
      kl25bpsvi = np.copy(kl25)
      kl75bpsvi = np.copy(kl75)

  fig.line(np.arange(kl50.shape[0]), kl50, color=alg[2], legend_label=alg[1], line_width=10)
  fig.line(np.arange(kl25.shape[0]), kl25, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
  fig.line(np.arange(kl75.shape[0]), kl75, color=alg[2], legend_label=alg[1], line_width=10, line_dash='dashed')
  # for computation time, don't show the coreset size 0 step since it's really really fast for all algs
  fig2.line(cput50[1:], kl50[1:], color=alg[2], legend_label=alg[1], line_width=10)
  fig2.patch(np.hstack((cput50[1:], cput50[1:][::-1])), np.hstack((kl75[1:], kl25[1:][::-1])), fill_color=alg[2], legend_label=alg[1], alpha=0.3)

  if alg[0] != 'PRIOR':
    fig3.line(csz50, kl50, color=alg[2], legend_label=alg[1], line_width=10)
    fig3.patch(np.hstack((csz50, csz50[::-1])), np.hstack((kl75, kl25[::-1])), fill_color=alg[2], legend_label=alg[1], alpha=0.3)
    fig3.legend.location='top_left'

for f in [fig, fig3]:
  f.legend.location='top_left'
  f.legend.label_text_font_size= '60pt'
  f.legend.glyph_width=50
  f.legend.glyph_height=50
  f.legend.spacing=10
  f.legend.visible = True
for f in [fig2]:
  f.legend.location='bottom_center'
  f.legend.label_text_font_size= '60pt'
  f.legend.glyph_width=50
  f.legend.glyph_height=50
  f.legend.spacing=10
  f.legend.visible = True

export_png(fig, filename=os.path.join(fldr_figs, dnm+"_KLDvsit.png"), height=1500, width=2000)
export_png(fig2, filename=os.path.join(fldr_figs, dnm+"_KLDvscput.png"), height=1500, width=2000)
export_png(fig3, filename=os.path.join(fldr_figs, dnm+"_KLDvssz.png"), height=1500, width=2000)

bkp.show(bkl.gridplot(figs))
