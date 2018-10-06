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
from pybold.hrf_model import spm_hrf, basis3_hrf
from pybold.bold_signal import (scale_factor_hrf_estimation,
                                basis3_hrf_estimation)
from pybold.utils import fwhm, tp


###############################################################################
# results management
print(__doc__)

d = datetime.now()
dirname = ('results_hrf_'
           'estimation_#{0}{1}{2}{3}{4}{5}'.format(d.year,
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
hrf_dur = 30.0
tr = 1.0
snr = 1.0

###############################################################################
# Estimate the HRF with time scale factor SPM HRF
true_hrf_delta = 1.5
sf_orig_hrf, t_hrf = spm_hrf(tr=tr, delta=true_hrf_delta, dur=hrf_dur)

params = {'dur': dur, 'tr': tr, 'hrf': sf_orig_hrf, 'nb_events': 5,
          'avg_dur': 1, 'std_dur': 3, 'overlapping': False, 'snr': snr,
          'random_state': 9}
noisy_ar_s, _, ai_s, _, t, _, _ = gen_rnd_bloc_bold(**params)

print("time scale factor SPM HRF estimation")
params = {'ai_i_s': ai_s, 'ar_s': noisy_ar_s, 'tr': tr, 'dur': hrf_dur,
          'verbose': 3, }

t0 = time.time()
sf_est_hrf, sf_J = scale_factor_hrf_estimation(**params)
sf_delta_t = np.round(time.time() - t0, 1)
sf_runtimes = np.linspace(0, sf_delta_t, len(sf_J))

print("Duration: {0} s".format(sf_delta_t))

###############################################################################
# Estimate the HRF 3 basis SPM HRF
hrf_params = np.array([0.6, 0.5, 0.2])
b3_orig_hrf, t_hrf = basis3_hrf(hrf_params, tr=tr, dur=hrf_dur)

params = {'dur': dur, 'tr': tr, 'hrf': b3_orig_hrf, 'nb_events': 5,
          'avg_dur': 1, 'std_dur': 3, 'overlapping': False, 'snr': snr,
          'random_state': 9}
noisy_ar_s, _, ai_s, _, t, _, _ = gen_rnd_bloc_bold(**params)

print("HRF 3 basis SPM HRF estimation")
params = {'ai_i_s': ai_s, 'ar_s': noisy_ar_s, 'tr': tr, 'dur': hrf_dur,
          'verbose': 3}

t0 = time.time()
b3_est_hrf, b3_J = basis3_hrf_estimation(**params)
b3_delta_t = np.round(time.time() - t0, 1)
b3_runtimes = np.linspace(0, b3_delta_t, len(sf_J))

print("Duration: {0} s".format(b3_delta_t))

###############################################################################
# plotting scale factor SPM HRF
print("Results directory: '{0}'".format(dirname))

# plot 1
fig = plt.figure(np.random.randint(9999), figsize=(20, 10))

label = "Orig. HRF, FWHM={0:.2f}s".format(fwhm(t_hrf, sf_orig_hrf))
plt.plot(t_hrf, sf_orig_hrf, '-b', label=label, lw=3.0)
label = ("Est. scale factor SPM HRF, FWHM={0:.2f}s, TP={1:.2f}s"
         "".format(fwhm(t_hrf, sf_est_hrf), tp(t_hrf, sf_est_hrf)))
plt.plot(t_hrf, sf_est_hrf, '-*r', label=label, lw=3.0)
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.legend()
plt.title("Original HRF", fontsize=20)

filename = "sf_est_hrf.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 2
fig = plt.figure(np.random.randint(9999), figsize=(16, 8))

plt.plot(t, noisy_ar_s, '-b', label="Noisy BOLD signal", linewidth=1.0)
plt.plot(t, ai_s, '-g', label="Block signal", linewidth=1.0)

plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.ylim(-2.0, 2.0)
plt.legend()
plt.title("Input signals, TR={0}s".format(tr), fontsize=20)

filename = "sf_bold_signal.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 3
fig = plt.figure(np.random.randint(9999), figsize=(20, 10))
plt.plot(sf_runtimes, sf_J, label="scale factor SPM HRF", linewidth=3.0)
plt.xlabel("times (s)")
plt.ylabel("cost function")
plt.legend()
plt.title("Evolution of the cost function")

filename = "sf_cost_function.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

###############################################################################
# plotting 3 basis SPM HRF
print("Results directory: '{0}'".format(dirname))

# plot 1
fig = plt.figure(np.random.randint(9999), figsize=(20, 10))

label = "Orig. HRF, FWHM={0:.2f}s".format(fwhm(t_hrf, b3_orig_hrf))
plt.plot(t_hrf, b3_orig_hrf, '-b', label=label, lw=3.0)
label = ("Est. 3 basis SPM HRF, FWHM={0:.2f}s, TP={1:.2f}s"
         "".format(fwhm(t_hrf, b3_est_hrf), tp(t_hrf, b3_est_hrf)))
plt.plot(t_hrf, b3_est_hrf, '-*r', label=label, lw=3.0)
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.legend()
plt.title("Original HRF", fontsize=20)

filename = "b3_est_hrf.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 2
fig = plt.figure(np.random.randint(9999), figsize=(16, 8))

plt.plot(t, noisy_ar_s, '-b', label="Noisy BOLD signal", linewidth=1.0)
plt.plot(t, ai_s, '-g', label="Block signal", linewidth=1.0)

plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.ylim(-2.0, 2.0)
plt.legend()
plt.title("Input signals, TR={0}s".format(tr), fontsize=20)

filename = "b3_bold_signal.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 3
fig = plt.figure(np.random.randint(9999), figsize=(20, 10))
plt.plot(sf_runtimes, sf_J, label="scale factor SPM HRF", linewidth=3.0)
plt.xlabel("times (s)")
plt.ylabel("cost function")
plt.legend()
plt.title("Evolution of the cost function")

filename = "b3_cost_function.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
