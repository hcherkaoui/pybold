# coding: utf-8
""" Blind deconvolution example.
"""
import os
import shutil
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pybold.data import gen_event_bold
from pybold.hrf_model import gen_hrf_spm_dict_normalized, spm_hrf
from pybold.utils import fwhm, inf_norm
from pybold.bold_signal import sparse_encoding_hrf_blind_events_deconvolution


###############################################################################
# results management
print(__doc__)

d = datetime.now()
dirname = ('results_blind_deconvolution_sparse_encoding_hrf_'
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
dur = 10  # minutes
tr = 1.0
snr = 1.0

# True HRF
true_hrf_delta = 1.5
normalized_hrf = True
orig_hrf, t_hrf = spm_hrf(tr=tr, delta=true_hrf_delta)
len_hrf = len(orig_hrf)

# dict of HRF
hrf_dico = gen_hrf_spm_dict_normalized(tr=tr)

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
noisy_ar_s, ar_s, i_s, t, _, _ = gen_event_bold(**params)

###############################################################################
# blind deconvolution
init_hrf_delta = 1.0
init_hrf, _ = spm_hrf(tr=tr, delta=init_hrf_delta)
params = {'noisy_ar_s': noisy_ar_s,
          'tr': tr,
          'hrf_dico': hrf_dico,
          'lbda_bold': 1.0,
          'lbda_hrf': 1.0,
          'init_hrf': init_hrf,
          'hrf_fixed_ampl': False,
          'nb_iter': 100,
          'verbose': 1,
          }

t0 = time.time()
results = sparse_encoding_hrf_blind_events_deconvolution(**params)
est_ar_s, est_i_s, est_hrf, sparse_encoding_hrf, J = results
delta_t = time.time() - t0
runtimes = np.linspace(0, delta_t, len(J))
print("Duration: {0:.2f} s".format(delta_t))

###############################################################################
# post-processing
if True:
    est_ar_s, est_i_s, est_hrf = inf_norm([est_ar_s, est_i_s, est_hrf])
    ar_s, orig_hrf = inf_norm([ar_s, orig_hrf])

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
ax2.stem(t, est_i_s, '-g', label="Est. innovation signal")
ax2.stem(t, i_s, '-b', label="Orig. innovation signal")
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

label = "Orig. HRF (FWHM={0:.2f}s)".format(fwhm(t_hrf, orig_hrf))
plt.plot(t_hrf, orig_hrf, '-b', label=label)
label = ("Est. HRF (FWHM={0:.2f}s, |x|_1={1:.2f},"
         "sum_i x_i={2:.2f})".format(fwhm(t_hrf, est_hrf),
                                     np.sum(np.abs(sparse_encoding_hrf)),
                                     np.sum(sparse_encoding_hrf)))
plt.plot(t_hrf, est_hrf, '-*g', label=label)
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.legend()
plt.title("Original HRF, TR={0}s".format(tr), fontsize=20)

filename = "est_hrf_{0}.png".format(true_hrf_delta)
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 3
fig = plt.figure(3, figsize=(15, 10))

plt.stem(sparse_encoding_hrf, '-*b', label="Est. coef")
plt.xlabel("FWHM of the atoms")
plt.ylabel("ampl.")
plt.legend()
plt.title("Est. sparse encoding HRF\n(ordered from tighter to the larger)",
          fontsize=20)

filename = "sparse_encoding_hrf_{0}.png".format(true_hrf_delta)
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 4
fig = plt.figure(4, figsize=(20, 10))
plt.plot(runtimes, J)
plt.xlabel("times (s)")
plt.ylabel("cost function")
plt.title("Evolution of the cost function")

filename = "cost_function.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
