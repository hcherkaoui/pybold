# coding: utf-8
"""Simple deconvolution example.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from pybold.data import gen_random_events
from pybold.bold_signal import bold_deconvolution


print(__doc__)


###############################################################################
# generate data
dur = 5  # minutes
tr = 1.0
snr = 1.0
params = {'dur': dur,
          'tr': tr,
          'hrf_time_length': 32.0,
          'nb_events': 6,
          'avg_dur': 1,
          'std_dur': 8,
          'overlapping': False,
          'snr': snr,
          'random_state': 99,
          }

noisy_ar_s, ar_s, ai_s, _, t, _, _, _ = gen_random_events(**params)

###############################################################################
# deconvolve the signal
t0 = time.time()
est_ar_s, est_ai_s, est_i_s = bold_deconvolution(
                                 noisy_ar_s, tr=tr, hrf_time_length=32.0,
                                 lbda=5.0e-5
                                                )
delta_t = np.round(time.time() - t0, 3)
print("Duration: {0} s".format(delta_t))

###############################################################################
# plotting
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
ax2.stem(t, est_i_s, '-g', label="Est. innovation signal")
ax2.axhline(0.0)
ax2.set_xlabel("time (s)")
ax2.set_ylabel("ampl.")
ax2.legend()
ax2.set_title("Estimated signals, TR={0}s".format(tr), fontsize=15)

plt.tight_layout()

filename = "bold_signal.png"
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
