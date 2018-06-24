# coding: utf-8
"""Simple HRF estimation
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from pybold.data import gen_random_events
from pybold.bold_signal import hrf_sparse_encoding_estimation


print(__doc__)


###############################################################################
# generate data
dur = 5  # minutes
tr = 0.5
hrf_time_length = 50.
params = {'dur': dur,
          'tr': tr,
          'hrf_time_length': hrf_time_length,
          'nb_events': 10,
          'avg_dur': 1,
          'std_dur': 5,
          'overlapping': False,
          'random_state': 0,
          }
_, ar_s, ai_s, _, t, orig_hrf, _, _ = gen_random_events(**params)

###############################################################################
# Estimate the HRF.
t0 = time.time()
est_hrf, sparse_encoding_hrf = hrf_sparse_encoding_estimation(
                                    ai_s, ar_s, tr, nb_time_deltas=20,
                                    nb_onsets=0, lbda=1.0e-4)
delta_t = np.round(time.time() - t0, 1)

print("Duration: {0} s".format(delta_t))

###############################################################################
# plotting

# plot 0
fig = plt.figure(0, figsize=(20, 10))

plt.stem(sparse_encoding_hrf, label="Coef")
plt.xlabel("atoms")
plt.ylabel("ampl.")
plt.legend()
plt.title("Coef", fontsize=20)

filename = "coef_hrf_{0}.png".format(hrf_time_length)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
# plot 1
fig = plt.figure(1, figsize=(20, 10))

plt.plot(orig_hrf, '-b', label="Orignal HRF", linewidth=2.0)
plt.plot(est_hrf, '-*g', label="Estimated HR", linewidth=2.0)
plt.xlabel("scans")
plt.ylabel("ampl.")
plt.legend()
plt.title("Original HRF", fontsize=20)

filename = "est_hrf_{0}.png".format(hrf_time_length)
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
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
