# coding: utf-8
"""Simple deconvolution example.
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
from pybold.data import gen_regular_bloc_bold
from pybold.hrf_model import spm_hrf
from pybold.bold_signal import deconvolution


###############################################################################
# results management
print(__doc__)

d = datetime.now()
dirname = ('results_deconvolution_'
           '#{0}{1}{2}{3}{4}{5}'.format(d.year,
                                        d.month,
                                        d.day,
                                        d.hour,
                                        d.minute,
                                        d.second))
if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

###############################################################################
# generate data
hrf_dur = 30.
dur = 10  # minutes
tr = 1.0
snr = 1.0

# True HRF
true_hrf_delta = 1.5
orig_hrf, t_hrf = spm_hrf(tr=tr, delta=true_hrf_delta, dur=hrf_dur)
params = {'tr': tr,
          'dur': dur,
          'snr': snr,
          'hrf': orig_hrf,
          'random_state': 0,
          }
noisy_ar_s, ar_s, ai_s, i_s, t, _, noise = gen_regular_bloc_bold(**params)

###############################################################################
# deconvolve the signal
nb_iter = 500 if not is_travis else 1
params = {'noisy_ar_s': noisy_ar_s,
          'tr': tr,
          'hrf': orig_hrf,
          'lbda': None,
          'L2_res': True,
          'nb_iter': nb_iter,
          'verbose': 1,
          }

t0 = time.time()
est_ar_s, est_ai_s, est_i_s, J, R, G = deconvolution(**params)
delta_t = np.round(time.time() - t0, 3)
runtimes = np.linspace(0, delta_t, len(R))

print("Duration: {0} s".format(delta_t))

est_noise = noisy_ar_s - est_ar_s
print("noise std: {0:0.6f}, est. noise std: {1:0.6f}".format(np.std(noise), np.std(est_noise)))
print("noise L2: {0:0.6f}, est. noise L2: {1:0.6f}".format(np.linalg.norm(noise), np.linalg.norm(est_noise)))

###############################################################################
# plotting
print("Results directory: '{0}'".format(dirname))

# plot 1
fig = plt.figure(1, figsize=(20, 10))

# axis 1
ax0 = fig.add_subplot(3, 1, 1)
label = "Noisy activation related signal, snr={0}dB".format(snr)
ax0.plot(t, noisy_ar_s, '-y', label=label, linewidth=1.0)
ax0.plot(t, est_ar_s, '-g', label="Est. activation related signal",
         linewidth=1.0)
ax0.axhline(0.0, c='k')
ax0.set_xlabel("time (s)")
ax0.set_ylabel("ampl.")
ax0.legend()
ax0.set_title("Input noisy BOLD signals, TR={0}s".format(tr), fontsize=15)

# axis 1
ax1 = fig.add_subplot(3, 1, 2)
label = "Orig. activation related signal"
ax1.plot(t, ar_s, '-b', label=label, linewidth=1.0)
ax1.plot(t, est_ar_s, '-g', label="Est. activation related signal",
         linewidth=1.0)
ax1.axhline(0.0, c='k')
ax1.set_xlabel("time (s)")
ax1.set_ylabel("ampl.")
ax1.legend()
ax1.set_title("Estimated convolved signals, TR={0}s".format(tr),
              fontsize=15)

# axis 2
ax2 = fig.add_subplot(3, 1, 3)
label = "Orig. activation inducing signal, snr={0}dB".format(snr)
ax2.plot(t, ai_s, '-b', label=label, linewidth=1.0)
ax2.plot(t, est_ai_s, '-g', label="Est. activation inducing signal",
         linewidth=1.0)
ax2.stem(t, est_i_s, '-g', label="Est. innovation signal")
ax2.axhline(0.0, c='k')
ax2.set_xlabel("time (s)")
ax2.set_ylabel("ampl.")
ax2.legend()
ax2.set_title("Estimated signals, TR={0}s".format(tr), fontsize=15)

plt.tight_layout()

filename = "bold_signal.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 2
fig = plt.figure(2, figsize=(5, 5))
plt.plot(runtimes, J, linewidth=3.0)
plt.xlabel("times (s)")
plt.ylabel("residual")
plt.title("Evolution of the residual")

filename = "residual.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
