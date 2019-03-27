# coding: utf-8
"""Simple HRF estimation
"""
import os
is_travis = ('TRAVIS' in os.environ)
if is_travis:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'

import shutil
import time
from datetime import datetime
import numpy as np
from pybold.data import gen_rnd_bloc_bold
from pybold.hrf_model import spm_hrf
from pybold.bold_signal import hrf_estim as he
from pybold.utils import inf_norm


###############################################################################
# results management
print(__doc__)

d = datetime.now()
dirname = ('results_hrf_est_#{0}{1}{2}'.format(d.hour, d.minute, d.second))

if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

###############################################################################
# experimentation parameters
dur = 3.5
hrf_dur = 20.0
TR = 0.75
snr = 5.0
random_state = 99

orig_hrf, t_hrf = spm_hrf(1.0, t_r=TR, dur=hrf_dur, normalized_hrf=False)
params = {'dur': dur, 'tr': TR, 'hrf': orig_hrf, 'nb_events': 2,
          'avg_dur': 12, 'std_dur': 0.0, 'overlapping': False, 'snr': snr,
          'random_state': random_state}
y_tilde, y, z, _, t, _, _ = gen_rnd_bloc_bold(**params)

###############################################################################
# HRF estimation
params = {'z': z, 'y': y_tilde, 't_r': TR, 'dur': hrf_dur, 'verbose': 3}
t0 = time.time()
est_hrf, J = he(**params)
delta_t = time.time() - t0
runtimes = np.linspace(0.0, delta_t, len(J))

print("Duration: {0} s".format(delta_t))
print("Results directory: '{0}'".format(dirname))

###############################################################################
# plotting
fig = plt.figure("HRF", figsize=(8, 4))
label = "Est. SPM HRF"
plt.plot(t_hrf, orig_hrf, '-b', label="Orig. HRF", lw=3.0)
plt.plot(t_hrf, est_hrf, '--r', label="Est. HRF", lw=3.0)
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.legend(fontsize=15, framealpha=0.3)
plt.title("HRF recovery", fontsize=15)
filename = "est_hrf.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

fig = plt.figure("Signals", figsize=(8, 4))
label = "Normalized noisy BOLD signal"
plt.plot(t, inf_norm(y_tilde), '-y', label=label, lw=3.0)
plt.plot(t, z, '-k', label="Block signal", lw=3.0)
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.ylim(-2.0, 2.0)
plt.legend(fontsize=15, framealpha=0.3)
plt.title("Input signals, TR={0}s".format(TR), fontsize=15)
filename = "bold_signal.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

fig = plt.figure("Cost-function", figsize=(5, 5))
plt.plot(runtimes, J, lw=3.0)
plt.xlabel("times (s)")
plt.ylabel("cost-function")
plt.title("Evolution of the cost-function", fontsize=15)
filename = "cost_function.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
