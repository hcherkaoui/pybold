# coding: utf-8
"""Simple HRF estimation
"""
import os
import shutil
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pybold.data import gen_event_bold
from pybold.hrf_model import spm_hrf
from pybold.bold_signal import hrf_scale_factor_estimation
from pybold.utils import fwhm


###############################################################################
# results management
print(__doc__)

d = datetime.now()
dirname = ("results_hrf_scale_factor_estimation_"
           "#{0}{1}{2}{3}{4}{5}".format(d.year,
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
dur = 10  # minutes
tr = 1.0
snr = 1.0

# True HRF
true_delta = 1.0
dur = 60.0
orig_hrf, t_hrf = spm_hrf(delta=true_delta, tr=tr, dur=dur)

# data generation
params = {'dur': dur,
          'tr': tr,
          'hrf': orig_hrf,
          'nb_events': 5,
          'avg_ampl': 1,
          'std_ampl': 3,
          'snr': snr,
          'random_state': 9,
          }
noisy_ar_s, _, i_s, t, _, _ = gen_event_bold(**params)


###############################################################################
# Estimate the HRF
params = {'ai_i_s': i_s,
          'ar_s': noisy_ar_s,
          'tr': tr,
          'dur': dur,
          'verbose': 3,
          }
t0 = time.time()
est_hrf, J = hrf_scale_factor_estimation(**params)
delta_t = np.round(time.time() - t0, 1)
runtimes = np.linspace(0, delta_t, len(J))
print("Duration: {0} s".format(delta_t))

###############################################################################
# plotting

# plot 0
fig = plt.figure(0, figsize=(20, 10))

label = "Orig. HRF, FWHM={0:.2f}s".format(fwhm(t_hrf, orig_hrf))
plt.plot(orig_hrf, '-b', label=label, linewidth=2.0)
label = "Est. HRF, FWHM={0:.2f}s".format(fwhm(t_hrf, est_hrf))
plt.plot(est_hrf, '--g', label=label, linewidth=2.0)
plt.xlabel("scans")
plt.ylabel("ampl.")
plt.legend()
plt.title("Original HRF", fontsize=20)

filename = "est_hrf.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 1
fig = plt.figure(1, figsize=(16, 8))

plt.plot(t, noisy_ar_s, '-b', label="Noisy BOLD signal", linewidth=2.0)
plt.stem(t, i_s, '-g', label="Block signal", linewidth=2.0)

plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.ylim(-2.0, 2.0)
plt.legend()
plt.title("Input signals, TR={0}s".format(tr), fontsize=20)

filename = "bold_signal.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 2
fig = plt.figure(2, figsize=(20, 10))
plt.plot(runtimes, J)
plt.xlabel("times (s)")
plt.ylabel("cost function")
plt.title("Evolution of the cost function")

filename = "cost_function.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
