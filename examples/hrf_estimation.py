# coding: utf-8
"""Simple HRF estimation
"""
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pybold.data import gen_random_events, gen_hrf_spm_dict, spm_hrf
from pybold.bold_signal import (hrf_sparse_encoding_estimation,
                                sparse_hrf_ampl_corr)


print(__doc__)


###############################################################################
# generate data
dur = 10  # minutes
tr = 0.5

# True HRF
true_hrf_time_length = 50.0
orig_hrf, _, _ = spm_hrf(tr, time_length=true_hrf_time_length)

# dict of HRF
nb_time_deltas = 500
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
          'nb_events': 20,
          'avg_dur': 1,
          'std_dur': 5,
          'overlapping': True,
          'unitary_block': True,
          'random_state': 0,
          }
_, ar_s, ai_s, _, t, _, _, _ = gen_random_events(**params)


###############################################################################
# Estimate the HRF
t0 = time.time()
est_hrf, sparse_encoding_hrf = hrf_sparse_encoding_estimation(
                                    ai_s, ar_s, tr, hrf_dico, lbda=1.0e-4)
delta_t = np.round(time.time() - t0, 1)
print("Duration: {0} s".format(delta_t))

###############################################################################
# re-estimation of the amplitude
est_hrf, sparse_encoding_hrf = sparse_hrf_ampl_corr(sparse_encoding_hrf, ar_s,
                                                    hrf_dico, ai_s)

###############################################################################
# plotting
d = datetime.now()
dirname = 'results_hrf_estimation_{0}_{1}_{2}_{3}_{4}'.format(d.year,
                                                              d.month,
                                                              d.day,
                                                              d.hour,
                                                              d.minute)
if not os.path.exists(dirname):
    os.makedirs(dirname)

# plot 0
fig = plt.figure(0, figsize=(20, 10))
plt.stem(true_sparse_encoding_hrf, '-*r', label="Coef")
plt.stem(sparse_encoding_hrf, label="Coef")
plt.xlabel("atoms")
plt.ylabel("ampl.")
plt.legend()
plt.title("Coef", fontsize=20)

filename = "coef_hrf_{0}.png".format(true_hrf_time_length)
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
# plot 1
fig = plt.figure(1, figsize=(20, 10))

plt.plot(orig_hrf, '-b', label="Orignal HRF", linewidth=2.0)
plt.plot(est_hrf, '--g', label="Estimated HR", linewidth=2.0)
plt.xlabel("scans")
plt.ylabel("ampl.")
plt.legend()
plt.title("Original HRF", fontsize=20)

filename = "est_hrf_{0}.png".format(true_hrf_time_length)
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 3
fig = plt.figure(3, figsize=(16, 8))

plt.plot(t, ar_s, '-b', label="Denoised BOLD signal",
         linewidth=2.0)
plt.plot(t, ai_s, '-g', label="Block signal",
         linewidth=2.0)

plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.ylim(-2.0, 2.0)
plt.legend()
plt.title("Input signals, TR={0}s".format(tr), fontsize=20)

filename = "bold_signal.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
