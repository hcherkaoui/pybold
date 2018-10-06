# coding: utf-8
""" Blind deconvolution example.
"""
import os
is_travis = ('TRAVIS' in os.environ)
if is_travis:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'

import shutil
import time
from datetime import datetime
import pickle
import numpy as np
from pybold.data import gen_regular_bloc_bold
from pybold.hrf_model import basis3_hrf, spm_hrf
from pybold.utils import fwhm, inf_norm
from pybold.bold_signal import blind_deconvolution


# 'lbda_bold': 0.270,  # SNR 1dB
# 'lbda_bold': 0.140,  # SNR 5dB
# 'lbda_bold': 0.110,  # SNR 8dB
# 'lbda_bold': 0.100,  # SNR 10dB
# 'lbda_bold': 0.055,  # SNR 20dB
# 'lbda_bold': 0.018,  # SNR 100dB

###############################################################################
# results management
print(__doc__)

date = datetime.now()
dirname = ('results_blind_deconvolution_'
           '#{0}{1}{2}{3}{4}{5}'.format(date.year,
                                        date.month,
                                        date.day,
                                        date.hour,
                                        date.minute,
                                        date.second))

if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

###############################################################################
# generate data
hrf_dur = 30.
dur = 10  # minutes
TR = 2.0
snr = 8.0

delta_orig = 1.0
orig_hrf, t_hrf = spm_hrf(delta_orig, tr=TR, dur=hrf_dur)
params = {'tr': TR,
          'dur': dur,
          'snr': snr,
          'hrf': orig_hrf,
          'random_state': 0,
          }
noisy_ar_s, ar_s, ai_s, i_s, t, _, _ = gen_regular_bloc_bold(**params)

###############################################################################
# blind deconvolution
hrf_params_init = np.array([0.5, 0.5, 0.1])
init_hrf, _ = basis3_hrf(hrf_basis3_params=hrf_params_init, tr=TR, dur=hrf_dur)

nb_iter = 50 if not is_travis else 1

params = {'noisy_ar_s': noisy_ar_s,
          'tr': TR,
          'lbda': 5.0,
          'L2_res': True,
          'hrf_func': basis3_hrf,
          'hrf_params': hrf_params_init,
          'hrf_cst_params': [TR, hrf_dur],
          'bounds': [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
          'dur_hrf': hrf_dur,
          'nb_iter': nb_iter,
          'verbose': 1,
          'plotting': True,
          'ai_s': ai_s,
          'hrf': orig_hrf,
          }

t0 = time.time()
results = blind_deconvolution(**params)
est_ar_s, est_ai_s, est_i_s, est_hrf, d = results
delta_t = time.time() - t0

print("Duration: {0:.2f} s".format(delta_t))

###############################################################################
# post-processing
est_ar_s, est_ai_s, est_i_s = inf_norm([est_ar_s, est_ai_s, est_i_s])
orig_hrf, init_hrf, est_hrf = inf_norm([orig_hrf, init_hrf, est_hrf])
ar_s, ai_s = inf_norm([ar_s, ai_s])

###############################################################################
# archiving results
res = {'est_ar_s': est_ar_s,
       'est_ai_s': est_ai_s,
       'est_i_s': est_i_s,
       'est_hrf': est_hrf,
       'd': d,
       'noisy_ar_s': noisy_ar_s,
       'ar_s': ar_s,
       'ai_s': ai_s,
       'i_s': i_s,
       't': t,
       }

filename = os.path.join(dirname, "results.pkl")
print("Pickling results under '{0}'".format(filename))
with open(filename, "wb") as pfile:
    pickle.dump(res, pfile)

###############################################################################
# plotting
print("Results directory: '{0}'".format(dirname))

# plot 1
fig = plt.figure(1, figsize=(20, 10))

# axis 1
ax0 = fig.add_subplot(3, 1, 1)
label = "Noisy activation related signal, snr={0}dB".format(snr)
ax0.plot(t, noisy_ar_s, '-y', label=label, linewidth=1.0)
ax0.plot(t, est_ar_s, '-g', label="Est. activation related signal",
         linewidth=1.0)
ax0.axhline(0.0)
ax0.set_xlabel("time (s)")
ax0.set_ylabel("ampl.")
ax0.legend()
ax0.set_title("Input noisy BOLD signals, TR={0}s".format(TR), fontsize=15)

# axis 1
ax1 = fig.add_subplot(3, 1, 2)
label = "Orig. activation related signal"
ax1.plot(t, ar_s, '-b', label=label, linewidth=1.0)
ax1.plot(t, est_ar_s, '-g', label="Est. activation related signal",
         linewidth=1.0)
ax1.axhline(0.0)
ax1.set_xlabel("time (s)")
ax1.set_ylabel("ampl.")
ax1.legend()
plt.grid()
ax1.set_title("Estimated convolved signals, TR={0}s".format(TR),
              fontsize=15)

# axis 2
ax2 = fig.add_subplot(3, 1, 3)
label = "Orig. activation inducing signal, snr={0}dB".format(snr)
ax2.plot(t, ai_s, '-b', label=label, linewidth=1.0)
ax2.plot(t, est_ai_s, '-g', label="Est. activation inducing signal",
         linewidth=1.0)
ax2.stem(t, est_i_s, '-g', label="Est. innovation signal")
ax2.axhline(0.0)
ax2.set_xlabel("time (s)")
ax2.set_ylabel("ampl.")
ax2.legend()
plt.grid()
ax2.set_title("Estimated signals, TR={0}s".format(TR), fontsize=15)

plt.tight_layout()

filename = "bold_signal_blind.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 2
fig = plt.figure(2, figsize=(15, 10))
t_hrf = np.linspace(0, len(orig_hrf) * TR, len(orig_hrf))

label = "Orig. HRF, FWHM={0:.2f}s".format(fwhm(t_hrf, orig_hrf))
plt.plot(t_hrf, orig_hrf, '-b', label=label, lw=3)
label = "Est. HRF, FWHM={0:.2f}s".format(fwhm(t_hrf, est_hrf))
plt.plot(t_hrf, est_hrf, '-*g', label=label, lw=3)
label = "Init. HRF, FWHM={0:.2f}s".format(fwhm(t_hrf, init_hrf))
plt.plot(t_hrf, init_hrf, '--r', label=label, lw=3)
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.legend()
plt.grid()
plt.title("Original HRF, TR={0}s".format(TR), fontsize=20)

filename = "est_hrf.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 3
fig = plt.figure(3, figsize=(5, 5))
plt.plot(d['J'], linewidth=6.0)
plt.xlabel("n iters")
plt.ylabel("cost function")
plt.grid()
plt.title("Evolution of the cost function")

filename = "cost_function.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 4
fig = plt.figure(4, figsize=(5, 5))
plt.plot(d['r'], linewidth=6.0)
plt.xlabel("n iters")
plt.ylabel("residual")
plt.grid()
plt.title("Evolution of the residual")

filename = "residualpng"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 5
fig = plt.figure(5, figsize=(5, 5))
plt.plot(d['g'], linewidth=6.0)
plt.xlabel("n iters")
plt.ylabel("L_1 norm")
plt.grid()
plt.title("Evolution of the L_1 norm")

filename = "l1_norm"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# plot 6
if 'err_ai_s' in d:
    fig = plt.figure(6, figsize=(5, 5))
    plt.plot(d['err_ai_s'], linewidth=6.0)
    plt.xlabel("n iters")
    plt.ylabel("blocs signal error")
    plt.grid()
    plt.title("Evolution of the blocs signal error")

    filename = "err_ai_s.png"
    filename = os.path.join(dirname, filename)
    print("Saving plot under '{0}'".format(filename))
    plt.savefig(filename)

# plot 7
if 'err_hrf' in d:
    fig = plt.figure(7, figsize=(5, 5))
    plt.plot(d['err_hrf'], linewidth=6.0)
    plt.xlabel("n iters")
    plt.ylabel("HRF error")
    plt.grid()
    plt.title("Evolution of the HRF error")

    filename = "err_hrf.png"
    filename = os.path.join(dirname, filename)
    print("Saving plot under '{0}'".format(filename))
    plt.savefig(filename)
