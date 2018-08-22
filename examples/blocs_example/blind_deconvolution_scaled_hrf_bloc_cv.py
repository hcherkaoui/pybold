# coding: utf-8
""" Blind deconvolution example.
"""
import os
import shutil
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pybold.data import gen_bloc_bold
from pybold.hrf_model import spm_hrf
from pybold.utils import fwhm, inf_norm
from pybold.bold_signal import scaled_hrf_blind_blocs_deconvolution_cv


###############################################################################
# results management
print(__doc__)

d = datetime.now()
dirname = ('results_blind_deconvolution_scaled_hrf_cv_'
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
hrf_dur = 80.
dur = 10  # minutes
tr = 1.0
snr = 1.0

# True HRF
true_hrf_delta = 1.5
orig_hrf, t_hrf = spm_hrf(tr=tr, delta=true_hrf_delta, dur=hrf_dur)

# data generation
params = {'dur': dur,
          'tr': tr,
          'hrf': orig_hrf,
          'nb_events': 5,
          'avg_dur': 1,
          'std_dur': 3,
          'overlapping': False,
          'snr': snr,
          'random_state': 9,
          }
noisy_ar_s, ar_s, ai_s, _, t, _, _ = gen_bloc_bold(**params)

###############################################################################
# blind deconvolution
init_hrf_delta = 1.0
init_hrf, _ = spm_hrf(tr=tr, delta=init_hrf_delta, dur=hrf_dur)
params = {'noisy_ar_s': noisy_ar_s,
          'tr': tr,
          't_hrf': t_hrf,
          'orig_hrf': orig_hrf,
          'init_delta': init_hrf_delta,
          'dur_hrf': hrf_dur,
          'nb_iter': 100,
          'n_jobs': -1,
          'verbose': 10,
          }

t0 = time.time()
params, results = scaled_hrf_blind_blocs_deconvolution_cv(**params)
est_ar_s, est_ai_s, est_i_s, est_hrf, J = results
delta_t = time.time() - t0
runtimes = np.linspace(0, delta_t, len(J))

print("Duration: {0:.2f} s".format(delta_t))
print("Best lbda: {0}".format(params['lbda_bold']))

###############################################################################
# post-processing
if True:
    est_ar_s, est_ai_s, est_i_s, est_hrf = inf_norm([est_ar_s, est_ai_s,
                                                     est_i_s, est_hrf])
    ar_s, ai_s, orig_hrf = inf_norm([ar_s, ai_s, orig_hrf])

###############################################################################
# plotting
print("Results directory: '{0}'".format(dirname))

# plot 1
fig = plt.figure(1, figsize=(20, 10))

# axis 1
ax0 = fig.add_subplot(3, 1, 1)
label = "Noisy activation related signal, snr={0}dB".format(snr)
ax0.plot(t, noisy_ar_s, '-y', label=label, linewidth=2.0)
ax0.axhline(0.0)
ax0.set_xlabel("time (s)")
ax0.set_ylabel("ampl.")
ax0.legend()
ax0.set_title("Input noisy BOLD signals, TR={0}s".format(tr), fontsize=15)

# axis 1
ax1 = fig.add_subplot(3, 1, 2)
label = "Orig. activation related signal"
ax1.plot(t, ar_s, '-b', label=label, linewidth=2.0)
ax1.plot(t, est_ar_s, '-g', label="Est. activation related signal",
         linewidth=2.0)
ax1.axhline(0.0)
ax1.set_xlabel("time (s)")
ax1.set_ylabel("ampl.")
ax1.legend()
ax1.set_title("Estimated convolved signals, TR={0}s".format(tr),
              fontsize=15)

# axis 2
ax2 = fig.add_subplot(3, 1, 3)
label = "Orig. activation inducing signal, snr={0}dB".format(snr)
ax2.plot(t, ai_s, '-b', label=label, linewidth=2.0)
ax2.plot(t, est_ai_s, '-g', label="Est. activation inducing signal",
         linewidth=2.0)
ax2.stem(t, est_i_s, '-g', label="Est. innovation signal")
ax2.axhline(0.0)
ax2.set_xlabel("time (s)")
ax2.set_ylabel("ampl.")
ax2.legend()
ax2.set_title("Estimated signals, TR={0}s".format(tr), fontsize=15)

plt.tight_layout()

filename = "bold_signal_blind.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 2
fig = plt.figure(2, figsize=(15, 10))
t_hrf = np.linspace(0, len(orig_hrf) * tr, len(orig_hrf))

label = "Orig. HRF, FWHM={0:.2f}s".format(fwhm(t_hrf, orig_hrf))
plt.plot(t_hrf, orig_hrf, '-b', label=label)
label = "Est. HRF, FWHM={0:.2f}s".format(fwhm(t_hrf, est_hrf))
plt.plot(t_hrf, est_hrf, '-*g', label=label)
label = "Init. HRF, FWHM={0:.2f}s".format(fwhm(t_hrf, init_hrf))
plt.plot(t_hrf, init_hrf, '--r', label=label)
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.legend()
plt.title("Original HRF, TR={0}s".format(tr), fontsize=20)

filename = "est_hrf_{0}.png".format(true_hrf_delta)
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 3
fig = plt.figure(4, figsize=(20, 10))
plt.plot(runtimes, J)
plt.xlabel("times (s)")
plt.ylabel("cost function")
plt.title("Evolution of the cost function")

filename = "cost_function.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
