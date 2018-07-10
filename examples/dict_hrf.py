# coding: utf-8
""" Simple HRF dictionary generation example.
"""
import matplotlib.pyplot as plt
from pybold.data import gen_hrf_spm_dict


print(__doc__)

###############################################################################
# HRF dict
tr = 0.1
nb_time_deltas = 20
hrf_dico, t_hrf, hrf_params = gen_hrf_spm_dict(tr=tr,
                                               nb_time_deltas=nb_time_deltas)

###############################################################################
# plotting
plt.figure(figsize=(20, 15))

for idx, hrf in enumerate(hrf_dico.T):
    label = r"$\Delta_t$ = {0} s".format(hrf_params[idx])
    plt.plot(t_hrf, hrf, label=label)

plt.legend()
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.title("HRF dictionary, TR={0}s".format(tr), fontsize=20)

filename = "hrf_dict_example.png"
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
