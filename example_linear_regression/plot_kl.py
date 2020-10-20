import bokeh.plotting as bkp
from bokeh.io import export_png
import pickle as pk
import numpy as np
import sys, os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *

n_trials=int(sys.argv[1])
plot_every=int(sys.argv[2])
fldr_figs=sys.argv[3]
prfx=sys.argv[4]

plot_reverse_kl = True
trials = np.arange(1, n_trials)
nms = [('BPSVI', 'PSVI', pal[-1]), ('SVI', 'SparseVI', pal[0]), ('RAND', 'Uniform', pal[3]), ('GIGAO','GIGA (Optimal)', pal[1]), ('GIGAR','GIGA (Realistic)', pal[2])]

#plot the KL figure
fig = bkp.figure(y_axis_type='log', plot_width=750, plot_height=750, x_axis_label='Coreset Size', y_axis_label=('Reverse KL' if plot_reverse_kl else 'Forward KL') )
preprocess_plot(fig, '32pt', False, True)

for i, nm in enumerate(nms):
  kl = []
  sz = []
  for t in trials:
    f = open(prfx+'/results_'+nm[0]+'_' + str(t)+'.pk', 'rb')
    res = pk.load(f) #res = (w, rklw, fklw)
    prior = open(prfx+'/results_PRIOR_' + str(t)+'.pk', 'rb')
    res_prior = pk.load(prior)
    f.close()
    if plot_reverse_kl:
      kl.append(res[1][::plot_every]/res_prior[1][::plot_every])
    else:
      kl.append(res[2][::plot_every]/res_prior[2][::plot_every])
    sz.append( np.array([w.shape[0] for w in res[0]])[::plot_every])
  x = np.percentile(sz, 50, axis=0)
  fig.line(x, np.percentile(kl, 50, axis=0), color=nm[-1], line_width=5, legend_label=nm[1])
  fig.patch(x = np.hstack((x, x[::-1])), y = np.hstack((np.percentile(kl, 75, axis=0), np.percentile(kl, 25, axis=0)[::-1])), color=nm[-1], fill_alpha=0.4, legend_label=nm[1])
postprocess_plot(fig, '70pt', location='top_right', glyph_width=40)
fig.legend.background_fill_alpha=0.
fig.legend.border_line_alpha=0.
fig.legend.visible = True

#bkp.show(fig)
if not os.path.exists(fldr_figs):
  os.mkdir(fldr_figs)
export_png(fig, filename=os.path.join(fldr_figs, "KLDvsCstSize.png"), height=1500, width=2000)
