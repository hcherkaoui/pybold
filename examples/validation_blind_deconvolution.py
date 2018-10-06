# coding: utf-8
"""Simple HRF estimation
"""
import os
import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Memory, Parallel, delayed
from pybold.data import gen_rnd_bloc_bold
from pybold.hrf_model import spm_hrf
from pybold.bold_signal import scale_factor_hrf_estimation, blind_deconvolution
from pybold.hrf_model import MIN_DELTA, MAX_DELTA
from pybold.utils import inf_norm


INIT_HRF_PARAMS = MAX_DELTA
PARAMS_BOUND = [(MIN_DELTA + 1.0e-1, MAX_DELTA - 1.0e-1)]


def create_result_dir(sufixe='', root_path='.'):
    """Return the path of a result directory."""
    d = datetime.now()
    dirname = 'results_{0}#{1}{2}{3}{4}'.format(sufixe, d.day, d.hour,
                                                d.minute, d.second)
    dirname = os.path.join(root_path, dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def _gen_voxel(hrf, t_r, snr, dur):
    """Generate one voxel and its corresponding block signal."""
    params = {'dur': dur, 'tr': t_r, 'hrf': hrf, 'nb_events': 10,
              'avg_dur': 4, 'std_dur': 2, 'overlapping': False, 'snr': snr,
              'random_state': None}
    voxel, _, block, _, _, _, _ = gen_rnd_bloc_bold(**params)
    return voxel, block


def _gen_voxels(n_voxels, hrf, t_r, snr, dur, n_jobs, verbose):
    """Generate one voxel and its corresponding block signal."""
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_gen_voxel)(
                                    hrf=hrf, t_r=t_r, snr=snr, dur=dur)
                                    for i in range(n_voxels))
    voxels, blocks = [], []
    for v, b in res:
        voxels.append(v)
        blocks.append(b)
    return np.vstack(voxels).T, np.vstack(blocks).T


gen_voxels_cached = Memory('./.cachedir').cache(_gen_voxels)


def _hrf_est_single_voxel(voxel, block, tr, dur, verbose):
    """Return the estimated HRF for a single voxel."""
    params = {'ai_i_s': block, 'ar_s': voxel, 'tr': tr, 'dur': dur,
              'verbose': 0}
    hrf, _ = scale_factor_hrf_estimation(**params)
    return hrf


def _hrf_est_voxels(voxels, blocks, t_r, dur, n_jobs, verbose):
    """Return the estimated HRF for a group voxel."""
    hrfs = Parallel(n_jobs=n_jobs, verbose=verbose)(
                                    delayed(_hrf_est_single_voxel)(
                                    voxel, block, tr, dur, verbose)
                                    for voxel, block
                                    in zip(voxels.T, blocks.T))
    return np.vstack(hrfs).T


hrf_est_voxels_cached = Memory('./.cachedir').cache(_hrf_est_voxels)


def __bbd(voxel, tr, lbda, hrf_dur, L2_res, nb_iter):
    """Private helper."""
    params = {'noisy_ar_s': voxel, 'tr': tr, 'lbda': lbda,
              'hrf_func': spm_hrf, 'hrf_params': INIT_HRF_PARAMS,
              'hrf_cst_params': [tr, hrf_dur], 'bounds': PARAMS_BOUND,
              'L2_res': L2_res, 'dur_hrf': hrf_dur, 'nb_iter': nb_iter,
              'wind': 2, 'tol': 1.0e-4, 'verbose': 0, 'plotting': False,
              }
    res = blind_deconvolution(**params)
    _, block, _, hrf, _ = res
    return inf_norm([block, hrf])


def _bbd(voxels, tr, lbda, hrf_dur, L2_res, nb_iter, n_jobs, verbose):
    """"Estimate the HRF without the experimental paradigm."""
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(__bbd)(
                                    tr=tr, voxel=voxel, lbda=lbda,
                                    L2_res=L2_res, hrf_dur=hrf_dur,
                                    nb_iter=nb_iter)
                                    for voxel in voxels.T)
    blocks, hrfs = [], []
    for b, h in res:
        blocks.append(b)
        hrfs.append(h)
    return np.vstack(blocks).T, np.vstack(hrfs).T


bbd_cached = Memory('./.cachedir').cache(_bbd)


def plot_signals(signals, additional_signals=None, title=None, save=False,
                 filename='signas.png', verbose=False):
    """ """
    fig, ax = plt.subplots()
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

    root_dirname = create_result_dir()

    tr = 0.75
    hrf_dur = 20.0
    orig_hrf_params = 0.7
    lbda = 0.85
    nb_iter = 20
    n_voxels = 10
    snr = 1.0
    dur_run = 3.0  # minutes
    n_jobs = 4
    orig_hrf, t_hrf = spm_hrf(orig_hrf_params, tr=tr, dur=hrf_dur)

    voxels, blocks = gen_voxels_cached(n_voxels=n_voxels, hrf=orig_hrf, t_r=tr,
                                       snr=snr, dur=dur_run, n_jobs=n_jobs,
                                       verbose=10)

    hrfs = hrf_est_voxels_cached(voxels=voxels, blocks=blocks, t_r=tr,
                                 dur=hrf_dur, n_jobs=n_jobs, verbose=10)
    orig_hrf = inf_norm(orig_hrf)
    hrfs = inf_norm(hrfs.T).T

    print("#" * 80)

    dirname = create_result_dir(sufixe='{0:0.4f}'.format(lbda),
                                root_path=root_dirname)

    plot_signals(hrfs, additional_signals=orig_hrf, title="HRF reference",
                 save=True, filename=os.path.join(dirname, 'hrfs.png'),
                 verbose=True)

    params = {'voxels': voxels, 'tr': tr, 'lbda': lbda, 'hrf_dur': hrf_dur,
              'L2_res': True, 'nb_iter': nb_iter, 'n_jobs': n_jobs,
              'verbose': 10,
              }

    with open(os.path.join(dirname, 'params.pkl'), 'wb') as pfile:
        pickle.dump(params, pfile)

    est_blocks, est_hrfs = bbd_cached(**params)
    results = {'est_blocks': est_blocks, 'est_hrfs': est_hrfs}

    plot_signals(est_hrfs, additional_signals=inf_norm(orig_hrf),
                 title="HRF estimation", save=True,
                 filename=os.path.join(dirname, 'est_hrfs.png'),
                 verbose=True)

    with open(os.path.join(dirname, 'results.pkl'), 'wb') as pfile:
        pickle.dump(results, pfile)
