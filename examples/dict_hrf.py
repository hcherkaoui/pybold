# coding: utf-8
""" Simple HRF dictionary generation example.
"""
import matplotlib.pyplot as plt
import numpy as np
from pybold.data import gen_hrf_spm_dict_normalized


print(__doc__)

###############################################################################
# HRF dict
tr = 1.0
nb_time_deltas = 20
hrf_dico = gen_hrf_spm_dict_normalized(tr=tr, nb_time_deltas=nb_time_deltas)
len_hrf = hrf_dico.shape[0]
t_hrf = np.linspace(0, len_hrf, int(len_hrf/tr))

###############################################################################
# plotting
for idx, hrf in enumerate(hrf_dico.T):
    plt.plot(t_hrf, hrf)

plt.legend()
plt.xlabel("time (s)")
plt.ylabel("ampl.")
plt.title("HRF dictionary, TR={0}s".format(tr), fontsize=20)

filename = "hrf_dict_example.png"
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
