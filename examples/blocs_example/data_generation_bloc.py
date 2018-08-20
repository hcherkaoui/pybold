# coding: utf-8
""" Data generation example.
"""
import matplotlib.pyplot as plt
from pybold.data import gen_bloc_bold
from pybold.hrf_model import spm_hrf
from pybold.utils import fwhm


###############################################################################
# generate the signal
tr = 1.0
snr = 1.0
dur_orig = 4  # minutes
hrf_delta = 1.0
hrf, t_hrf = spm_hrf(tr=tr, delta=hrf_delta)
hrf_fwhm = fwhm(t_hrf, hrf)
params = {'dur': dur_orig,
          'tr': tr,
          'hrf': hrf,
          'nb_events': 4,
          'avg_dur': 1,
          'std_dur': 4,
          'overlapping': False,
          'snr': snr,
          'random_state': 0,
          }

res = gen_bloc_bold(**params)
noisy_ar_s, ar_s, ai_s, i_s, t, _, noise = res

###############################################################################
# plotting
fig = plt.figure(1, figsize=(18, 10))

# axis 1
ax1 = fig.add_subplot(3, 1, 1)

label = "Noisy BOLD signal, SNR={0}dB".format(snr)
ax1.plot(t, noisy_ar_s, '-y', label=label, linewidth=2.0)
ax1.plot(t, ar_s, '-b', label="Denoised BOLD signal", linewidth=2.0)

ax1.set_xlabel("time (s)")
ax1.set_ylabel("ampl.")
ax1.legend()
ax1.set_title("Convolved signals, TR={0}s".format(tr), fontsize=20)

# axis 2
ax2 = fig.add_subplot(3, 1, 2)

ax2.plot(t, ai_s, '-r', label="Block signal", linewidth=2.0)
ax2.stem(t, i_s, '-g', label="Dirac source signal")

ax2.set_xlabel("time (s)")
ax2.set_ylabel("ampl.")
ax2.set_ylim(-1.5, 1.5)
ax2.legend()
ax2.set_title("Source signals, TR={0}s".format(tr), fontsize=20)

# axis 3
ax3 = fig.add_subplot(3, 1, 3)

ax3.plot(t_hrf, hrf, label="Original HRF")

ax3.set_xlabel("time (s)")
ax3.set_ylabel("ampl.")
ax3.legend()
title = r"HRF, TR={0}s, FWHM={1:.2f}s".format(tr, fwhm(t_hrf, hrf))
ax3.set_title(title, fontsize=20)

plt.tight_layout()

filename = "generation_data_bloc_tr_{0}.png".format(tr)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
