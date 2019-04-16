# coding: utf-8
"""Simple HRF estimation
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from joblib import Memory, Parallel, delayed
from pybold.data import gen_rnd_bloc_bold
from pybold.hrf_model import spm_hrf
from pybold.bold_signal import bd
from pybold.hrf_model import MIN_DELTA, MAX_DELTA
from pybold.utils import inf_norm


THETA_BOUND = [(MIN_DELTA + 1.0e-1, MAX_DELTA - 1.0e-1)]


def _gen_voxel(hrf, t_r, snr, dur, avg_dur=12, nb_events=5, std_dur=1):
    """Generate one voxel and its corresponding block signal."""
    params = {'dur': dur, 'tr': t_r, 'hrf': hrf, 'nb_events': nb_events,
              'avg_dur': avg_dur, 'std_dur': std_dur, 'overlapping': False,
              'snr': snr, 'random_state': None}
    voxel, _, block, _, _, _, _ = gen_rnd_bloc_bold(**params)
    return voxel, block


def _gen_voxels(n_voxels, hrf, t_r, snr, dur, avg_dur, nb_events, std_dur,
                n_jobs, verbose):
    """Generate one voxel and its corresponding block signal."""
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_gen_voxel)(
                                    hrf=hrf, t_r=t_r, snr=snr, dur=dur,
                                    avg_dur=avg_dur, nb_events=nb_events,
                                    std_dur=std_dur)
                                    for i in range(n_voxels))
    voxels, blocks = [], []
    for v, b in res:
        voxels.append(v)
        blocks.append(b)
    return np.vstack(voxels).T, np.vstack(blocks).T


gen_voxels_cached = Memory('./.cachedir').cache(_gen_voxels)


def __bd(y, t_r, lbda, hrf_dur, nb_iter):
    """Private helper."""
    params = {'y': y, 't_r': t_r, 'lbda': lbda, 'theta_0': MAX_DELTA,
              'hrf_dur': hrf_dur, 'bounds': THETA_BOUND, 'nb_iter': nb_iter}
    _, block, _, hrf, _ = bd(**params)
    return inf_norm([block, hrf])


def _bd(voxels, t_r, lbda, hrf_dur, nb_iter, n_jobs, verbose):
    """"Estimate the HRF without the experimental paradigm."""
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(__bd)(
                                    t_r=t_r, y=voxel, lbda=lbda,
                                    hrf_dur=hrf_dur, nb_iter=nb_iter)
                                    for voxel in voxels.T)
    blocks, hrfs = [], []
    for b, h in res:
        blocks.append(b)
        hrfs.append(h)
    return np.vstack(blocks).T, np.vstack(hrfs).T


bd_cached = Memory('./.cachedir').cache(_bd)


def plot_signals(signals, additional_signals=None, title=None, save=False,
                 filename='signas.png', verbose=False):
    """Plotting"""
    _, ax = plt.subplots()
    mean = np.mean(signals, axis=1)
    std = np.std(signals, axis=1)
    bd = (mean - std, mean + std)
    ax.fill_between(range(len(mean)), bd[0], bd[1], alpha=0.2,
                    color='blue')
    ax.plot(mean, color='blue', lw=3.0)
    if additional_signals is not None:
        ax.plot(additional_signals, '--g', lw=3.0)
    if isinstance(title, str):
        plt.title(title)
    if save:
        if verbose:
            print("Saving plot under '{0}'".format(filename))
            print("_" * 80)
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == '__main__':

    TR = 0.75
    hrf_dur = 20.0
    orig_hrf_params = 0.7
    dur_run = 3.0  # minutes
    l_snr = [1.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    n_voxels = 100
    avg_dur = 12
    std_dur = 1
    nb_events = 5

    # optimal value (already grid-search)
    l_lbda = [4.0, 3.0, 2.5, 1.7, 1.6, 1.5]
    nb_iter = 500
    n_jobs = 3

    orig_hrf, t_hrf = spm_hrf(orig_hrf_params, t_r=TR, dur=hrf_dur)

    h = len(orig_hrf)
    n = int(dur_run * TR)

    err_hrfs, err_blocks = {}, {}
    for lbda, snr in zip(l_lbda, l_snr):

        print('*'*80)
        print("Running blind-deconvolution for SNR={0}dB".format(snr))
        print('*'*80)

        voxels, blocks = gen_voxels_cached(
                                n_voxels=n_voxels, hrf=orig_hrf.flatten(),
                                t_r=TR, snr=snr, dur=dur_run, avg_dur=avg_dur,
                                nb_events=nb_events, std_dur=std_dur,
                                n_jobs=n_jobs, verbose=10)

        params = {'voxels': voxels, 't_r': TR, 'lbda': lbda,
                  'hrf_dur': hrf_dur, 'nb_iter': nb_iter, 'n_jobs': n_jobs,
                  'verbose': 10,
                  }

        est_blocks, est_hrfs = bd_cached(**params)

        r_orig_hrf = np.repeat(orig_hrf[:, None], n_voxels, axis=1)
        err_hrfs[snr] = np.linalg.norm(est_hrfs - r_orig_hrf, axis=0)
        err_hrfs[snr] /= np.linalg.norm(orig_hrf)
        err_blocks[snr] = np.linalg.norm(est_blocks - blocks, axis=0)
        err_blocks[snr] /= np.linalg.norm(blocks, axis=0)

    mean_err_hrfs = np.array([np.mean(err_hrfs[snr]) for snr in l_snr])
    std_err_hrfs = np.array([np.std(err_hrfs[snr]) for snr in l_snr])
    mean_err_blocks = np.array([np.mean(err_blocks[snr]) for snr in l_snr])
    std_err_blocks = np.array([np.std(err_blocks[snr]) for snr in l_snr])

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.set_xlabel("SNR [dB]", fontsize=20)
    ax1.set_ylabel("Relative L2 norm error", fontsize=20)
    plt.errorbar(l_snr, mean_err_hrfs, yerr=std_err_hrfs, color='red',
                 label="HRF error", linewidth=3.0, elinewidth=3.0)
    plt.errorbar(l_snr, mean_err_blocks, yerr=std_err_blocks, color='blue',
                 label="Z error", linewidth=3.0, elinewidth=3.0)
    plt.xticks(l_snr)
    plt.legend(fontsize=20)
    ax1.tick_params(labelsize=20)

    plt.tight_layout()
    d = datetime.now()
    filename = 'err_{0}{1}{2}.pdf'.format(d.hour, d.minute, d.second)
    print("Saving err evolution under {0}".format(filename))
    plt.savefig(filename, dpi=150)
