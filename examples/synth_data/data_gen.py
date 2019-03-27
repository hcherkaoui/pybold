# coding: utf-8
""" Data generation example.
"""
import os
is_travis = ('TRAVIS' in os.environ)
if is_travis:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from pybold.data import gen_rnd_bloc_bold
from pybold.hrf_model import spm_hrf
from pybold.utils import fwhm, tp


###############################################################################
# generate the signal
tr = 1.0
snr = 1.0
dur = 4  # minutes
hrf_dur = 30.0
hrf, t_hrf = spm_hrf(1.0, t_r=tr, dur=hrf_dur)
hrf_fwhm = fwhm(t_hrf, hrf)
hrf_tp = tp(t_hrf, hrf)
params = {'dur': dur,
          'tr': tr,
          'hrf': hrf,
          'nb_events': 4,
          'avg_dur': 1,
          'std_dur': 4,
          'overlapping': False,
          'snr': snr,
          'random_state': 0,
          }

res = gen_rnd_bloc_bold(**params)
noisy_ar_s, ar_s, ai_s, i_s, t, _, noise = res

###############################################################################
# plotting
fig = plt.figure(1, figsize=(15, 7))

# axis 1
ax1 = fig.add_subplot(3, 1, 1)

label = "Noisy BOLD signal, SNR={0}dB, TR={0}s".format(snr, tr)
ax1.plot(t, noisy_ar_s, '-y', label=label, lw=3)
ax1.plot(t, ar_s, '-b', label="Denoised BOLD signal, TR={0}s".format(tr), lw=3)
ax1.set_xlabel("time (s)")
ax1.set_ylabel("ampl.")
ax1.legend(fontsize=15, framealpha=0.3)

# axis 2
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(t, ai_s, '-r', label="Block signal", lw=3)
ax2.stem(t, i_s, '-g', label="Dirac source signal")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("ampl.")
ax2.set_ylim(-1.5, 1.5)
ax2.legend(fontsize=15, framealpha=0.3)
ax2.set_title("Source signals, TR={0}s".format(tr), fontsize=15)

# axis 3
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(t_hrf, hrf, label="Original HRF", lw=3)
ax3.set_xlabel("time (s)")
ax3.set_ylabel("ampl.")
ax3.legend(fontsize=15, framealpha=0.3)
title = (r"HRF, TR={0}s, FWHM={1:.2f}s, "
         "TP={2:.2f}s".format(tr, fwhm(t_hrf, hrf), tp(t_hrf, hrf)))
ax3.set_title(title, fontsize=15)

plt.tight_layout()

filename = "gen_data.png"
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
