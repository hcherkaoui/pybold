# coding: utf-8
""" Data generation example.
"""
import numpy as np
import matplotlib.pyplot as plt
from pybold.hrf_model import spm_hrf, il_hrf
from pybold.utils import fwhm


###############################################################################
# params
tr = 1.0
dur = 20.0
delta = 1.0
hrf_logit_params = np.array([1.521, -21.862, 20.339, 3.18, 9.009, 9.315, 1.934,
                             3.176, 3.131])

###############################################################################
# HRF
spm_hrf, t_hrf = spm_hrf(tr=tr, dur=dur, delta=delta)
spm_hrf_fwhm = fwhm(t_hrf, spm_hrf)

il_hrf, t_hrf, il_s = il_hrf(hrf_logit_params=hrf_logit_params, dur=dur, tr=tr)
il_hrf_fwhm = fwhm(t_hrf, il_hrf)

###############################################################################
# plotting
fig = plt.figure(1, figsize=(18, 10))

# axis 1
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(t_hrf, spm_hrf, linewidth=2.0)
ax1.set_xlabel("time (s)")
ax1.set_ylabel("ampl.")
ax1.legend()
ax1.set_title("spm HRF, TR={0}s, FWHM={1:.2f}s".format(tr, spm_hrf_fwhm))

# axis 2
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t_hrf, il_hrf, label="HRF", linewidth=2.0)
ax2.set_xlabel("time (s)")
ax2.set_ylabel("ampl.")
ax2.set_title("IL HRF, TR={0}s, FWHM={1:.2f}s".format(tr, il_hrf_fwhm))


filename = "generated_hrf_tr_{0}.png".format(tr)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
