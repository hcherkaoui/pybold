# coding: utf-8
""" Blind deconvolution example.
"""
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pybold.data import gen_random_events, gen_hrf_spm_dict, spm_hrf
from pybold.utils import max_min_norm
from pybold.bold_signal import bold_blind_deconvolution


print(__doc__)


###############################################################################
# generate data
dur = 10  # minutes
tr = 1.0
snr = 1.0

# True HRF
true_hrf_time_length = 40.0
orig_hrf, _, _ = spm_hrf(tr, time_length=true_hrf_time_length)

# dict of HRF
nb_time_deltas = 300
hrf_dico, _, hrf_lengths = gen_hrf_spm_dict(tr=tr,
                                            nb_time_deltas=nb_time_deltas)
# add the True HRF in the dict of HRF
idx = 0
hrf_dico = np.c_[hrf_dico[:, :idx], orig_hrf.T, hrf_dico[:, idx:]]
hrf_lengths.insert(idx, true_hrf_time_length)
true_sparse_encoding_hrf = np.zeros(hrf_dico.shape[1])
true_sparse_encoding_hrf[idx] = 1

# data generation
params = {'dur': dur,
          'tr': tr,
          'hrf': orig_hrf,
          'nb_events': 10,
          'avg_dur': 1,
          'std_dur': 5,
          'overlapping': True,
          'snr': snr,
          'random_state': 42,
          }
noisy_ar_s, ar_s, ai_s, _, t, _, _, _ = gen_random_events(**params)

###############################################################################
# post-processing
init_hrf, _, _ = spm_hrf(tr=tr, time_length=30.0)
params = {'noisy_ar_s': noisy_ar_s,
          'tr': tr,
          'hrf_dico': hrf_dico,
          'lbda_bold': 1.0e-5,
          'lbda_hrf': 3.0e-4,
          'init_hrf': init_hrf,
          'nb_iter': 50,
          'verbose': 1,
          }

t0 = time.time()
results = bold_blind_deconvolution(**params)
est_ar_s, est_ai_s, est_i_s, est_hrf, sparse_encoding_hrf, J = results
delta_t = np.round(time.time() - t0, 3)
runtimes = np.linspace(0, delta_t, len(J))
print("Duration: {0} s".format(delta_t))

###############################################################################
# post-processing
noisy_ar_s, ar_s, ai_s = max_min_norm([noisy_ar_s, ar_s, ai_s])
est_ar_s, est_ai_s, est_i_s = max_min_norm([est_ar_s, est_ai_s, est_i_s])
est_hrf, orig_hrf, init_hrf = max_min_norm([est_hrf, orig_hrf, init_hrf])

###############################################################################
# plotting
d = datetime.now()
dirname = 'results_blind_deconvolution_{0}_{1}_{2}_{3}_{4}'.format(d.year,
                                                                   d.month,
                                                                   d.day,
                                                                   d.hour,
                                                                   d.minute)
if not os.path.exists(dirname):
    os.makedirs(dirname)

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

plt.plot(t_hrf, orig_hrf, '-b', label="Orig. HRF")
plt.plot(t_hrf, est_hrf, '-*g', label="Est. HRF")
plt.plot(t_hrf, init_hrf, '--r', label="Init. HRF")
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.legend()
plt.title("Original HRF, TR={0}s".format(tr), fontsize=20)

filename = "est_hrf_{0}.png".format(true_hrf_time_length)
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 3
fig = plt.figure(3, figsize=(15, 10))

plt.stem(sparse_encoding_hrf, '-b', label="Est. HRF")
plt.xlabel("atoms")
plt.ylabel("ampl.")
plt.legend()
plt.title("Est. sparse encoding HRF\n(the atoms are not ordered)", fontsize=20)

filename = "sparse_encoding_hrf_{0}.png".format(true_hrf_time_length)
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
