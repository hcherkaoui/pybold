# coding: utf-8
"""Simple deconvolution example.
"""
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pybold.data import gen_random_events, spm_hrf
from pybold.bold_signal import bold_deconvolution, i_s_ampl_corr


print(__doc__)


###############################################################################
# generate data
dur = 5  # minutes
tr = 1.0
snr = 1.0
hrf_time_length = 32.0
hrf, _, _ = spm_hrf(tr=tr, time_length=hrf_time_length)
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
noisy_ar_s, ar_s, ai_s, _, t, _, _, _ = gen_random_events(**params)

###############################################################################
# deconvolve the signal
lbda = 1.0e-4
t0 = time.time()
est_ar_s, est_ai_s, est_i_s, J = bold_deconvolution(
                                                    noisy_ar_s, tr=tr, hrf=hrf,
                                                    lbda=lbda
                                                    )
delta_t = np.round(time.time() - t0, 3)
runtimes = np.linspace(0, delta_t, len(J))
print("Duration: {0} s".format(delta_t))

###############################################################################
# re-estimation of the amplitude
est_ar_s, est_ai_s, est_i_s = i_s_ampl_corr(
                                    est_i_s=est_i_s,
                                    noisy_ar_s=noisy_ar_s,
                                    hrf=hrf
                                            )

###############################################################################
# plotting
d = datetime.now()
dirname = 'results_deconvolution_{0}_{1}_{2}_{3}_{4}'.format(d.year,
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
ax0.axhline(0.0, c='k')
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
ax1.axhline(0.0, c='k')
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
fig = plt.figure(2, figsize=(20, 10))
plt.plot(runtimes, J)
plt.xlabel("times (s)")
plt.ylabel("cost function")
plt.title("Evolution of the cost function")

filename = "cost_function.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
