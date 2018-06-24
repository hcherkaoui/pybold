# coding: utf-8
""" Blind deconvolution example.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from pybold.data import gen_random_events
from pybold.utils import max_min_norm
from pybold.bold_signal import bold_blind_deconvolution


print(__doc__)


###############################################################################
# generate data
dur = 10  # minutes
tr = 0.5
snr = 1.0
params = {'dur': dur,
          'tr': tr,
          'hrf_time_length': 50.0,
          'nb_events': 10,
          'avg_dur': 1,
          'std_dur': 5,
          'overlapping': True,
          'snr': snr,
          'random_state': 99,
          }

noisy_ar_s, ar_s, ai_s, _, t, orig_hrf, _, _ = gen_random_events(**params)

###############################################################################
# post-processing
params = {'noisy_ar_s': noisy_ar_s,
          'tr': tr,
          'nb_time_deltas': 50,
          'nb_onsets': 4,
          'lbda_bold': 2.0e-5,
          'lbda_hrf': 3.0e-4,
          }

t0 = time.time()
results = bold_blind_deconvolution(**params)
delta_t = np.round(time.time() - t0, 3)
print("Duration: {0} s".format(delta_t))

est_ar_s, est_ai_s, est_i_s, est_hrf, sparse_encoding_hrf = results
noisy_ar_s, ar_s, ai_s = max_min_norm([noisy_ar_s, ar_s, ai_s])
est_ar_s, est_ai_s, est_i_s = max_min_norm([est_ar_s, est_ai_s, est_i_s])
est_hrf, orig_hrf = max_min_norm([est_hrf, orig_hrf])

###############################################################################
# plotting

# plot 1
fig = plt.figure(1, figsize=(20, 10))

# axis 1
ax0 = fig.add_subplot(3, 1, 1)
label = "Noisy activation related signal, snr={0}dB".format(snr)
ax0.plot(t, noisy_ar_s, '--y', label=label, linewidth=2.0)
ax0.axhline(0.0)
ax0.set_xlabel("time (s)")
ax0.set_ylabel("ampl.")
ax0.legend()
ax0.set_title("Input noisy BOLD signals, TR={0}s".format(tr), fontsize=15)

# axis 1
ax1 = fig.add_subplot(3, 1, 2)
label = "Orig. activation related signal"
ax1.plot(t, ar_s, '-g', label=label, linewidth=2.0)
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
ax2.plot(t, ai_s, '-r', label=label, linewidth=2.0)
ax2.plot(t, est_ai_s, '-g', label="Est. activation inducing signal",
         linewidth=2.0)
ax2.axhline(0.0)
ax2.set_xlabel("time (s)")
ax2.set_ylabel("ampl.")
ax2.legend()
ax2.set_title("Estimated signals, TR={0}s".format(tr), fontsize=15)

plt.tight_layout()

filename = "bold_signal.png"
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 2
fig = plt.figure(2, figsize=(15, 10))
t_hrf = np.linspace(0, len(orig_hrf) * tr, len(orig_hrf))

plt.plot(t_hrf, orig_hrf, '-b', label="Orig. HRF")
plt.plot(t_hrf, est_hrf, '--g', label="Est. HRF")
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.legend()
plt.title("Original HRF", fontsize=20)

filename = "est_hrf.png"
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 3
fig = plt.figure(3, figsize=(15, 10))

plt.stem(sparse_encoding_hrf, '-b')
plt.xlabel("atoms")
plt.ylabel("ampl.")
plt.title("Est. sparse encoding HRF", fontsize=20)

filename = "sparse_encoding_hrf.png"
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
