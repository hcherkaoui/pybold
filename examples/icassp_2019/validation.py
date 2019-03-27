# coding: utf-8
""" Regions selection on HCP data.
"""
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'

import matplotlib
matplotlib.use('Agg')

import shutil
import pickle
import numpy as np
from joblib import Parallel, delayed, Memory
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
from pybold.bold_signal import bd
from pybold.hrf_model import spm_hrf, MIN_DELTA, MAX_DELTA
from pybold.utils import inf_norm
from hrf_estimation.rank_one_ import glm as pglm
from hcp_builder.dataset import fetch_subject_list
from utils import (create_result_dir, get_hcp_fmri_fname, get_protocol_hcp,
                   mask_n_max, plot_trial_z_map, hrf_coef_to_hrf, TR_HCP,
                   N_SCANS_HCP, DUR_RUN_HCP)


# usefull functions

if not fetch_subject_list():
    def fetch_subject_list():
        """Fix troubles cause by bug/wrong usage with hcp_builder."""
        return [100206, 996782]


pglm_cached = Memory('./.cachedir').cache(pglm, ignore=['n_jobs', 'verbose'])


def _bd(voxels, lbda, hrf_dur, n_jobs, verbose=10):
    """Blind deconvolution"""
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(bd)(
                voxel, t_r=TR_HCP, lbda=lbda,
                bounds=[(MIN_DELTA + 1.0e-1, MAX_DELTA - 1.0e-1)],
                hrf_dur=hrf_dur, nb_iter=100)
                for voxel in voxels.T)
    return res


bd_cached = Memory('./.cachedir').cache(_bd, ignore=['n_jobs', 'verbose'])


# global variables

subject_id = 0  # other possible values: 546 99 1 0 (or more) (from HCP)
hrf_dur = 20.0
n_voxels_to_retain = 100
n_jobs = 3  # nb CPU to use
lbdas_cue = [1.0]  # list the value for lambda for task 'cue'
lbdas_rh = [3.5]  # list the value for lambda for task 'rh'
lbdas_lh = [3.0]  # list the value for lambda for task 'lf'
all_lbdas = [lbdas_rh, lbdas_lh, lbdas_cue]
trials = ['rh', 'lh', 'cue']  # other possible values: 'lf' 'rf'


# start main

root_dir = create_result_dir(sufixe='fast_hcp_validation')
print("Saving results under '{0}'".format(root_dir))

print("archiving '{0}' under '{1}'".format(__file__, root_dir))
shutil.copyfile(__file__, os.path.join(root_dir, __file__))

fmri_img, anat_img = get_hcp_fmri_fname(fetch_subject_list()[subject_id],
                                        anat_data=True)

canonical_hrf = inf_norm(spm_hrf(1.0, t_r=TR_HCP, dur=hrf_dur,
                         normalized_hrf=False)[0])

ref_hrfs, b_est_hrfs, b_est_blocks = {}, {}, {}
for trial, lbdas in zip(trials, all_lbdas):

    print('*'*80)
    print("Running validation on HCP dataset for trial '{0}'".format(trial))
    print('*'*80)
    subject_hcp_id = fetch_subject_list()[subject_id]
    trial_type, onset, p_e, _ = get_protocol_hcp(subject_hcp_id, trial)

    masker = NiftiMasker(t_r=TR_HCP, standardize=True)
    voxels = masker.fit_transform(fmri_img)

    n, v = voxels.shape

    hrfs_coef, coef = pglm_cached(trial_type, onset, TR_HCP, voxels,
                                  drifts=np.zeros((n, 1)),
                                  mode='r1glm', basis='3hrf',
                                  n_jobs=n_jobs, verbose=0,
                                  return_design_matrix=False)

    mask_roi = mask_n_max(np.abs(coef), n_voxels_to_retain).flatten()
    mask_roi_img = masker.inverse_transform(mask_roi.astype(float))
    voxels = voxels[:, mask_roi]

    ref_hrfs[trial] = hrf_coef_to_hrf(hrfs_coef[:, mask_roi], hrf_dur, TR_HCP)

    title = "{0} voxels region definition '{1}'".format(n_voxels_to_retain,
                                                        trial)
    filename = os.path.join(root_dir, 'region_{0}.png'.format(trial))
    plot_trial_z_map(mask_roi_img, title='Region {0}'.format(trial), th=0.0,
                     bg=anat_img, save=True, verbose=True, display_mode='z',
                     filename=filename)

    filename = os.path.join(root_dir, 'ref_hrfs_{0}.pkl'.format(trial))
    with open(filename, 'wb') as pfile:
        pickle.dump(ref_hrfs, pfile)

    for lbda in lbdas:

        lbda_dir = os.path.join(root_dir, "results_{0:0.3f}".format(lbda))
        if not os.path.exists(lbda_dir):
            os.makedirs(lbda_dir)

        res = bd_cached(voxels=voxels, lbda=lbda, hrf_dur=hrf_dur,
                        n_jobs=n_jobs, verbose=10)

        b_est_hrfs[trial] = np.vstack([h for _, _, _, h, _ in res]).T
        b_est_blocks[trial] = np.vstack([b for _, b, _, _, _ in res]).T

        n_b_est_blocks = inf_norm(b_est_blocks[trial].T).T
        n_voxels = inf_norm(voxels.T).T

        fig, ax = plt.subplots(figsize=(10, 5))

        mean = np.mean(n_voxels, axis=1)
        std = np.std(n_voxels, axis=1)
        t = np.arange(n_voxels.shape[0]) * TR_HCP
        borders = (mean - std, mean + std)
        ax.fill_between(t, borders[0], borders[1], alpha=0.2, color='black')
        ax.plot(t, mean, color='black', lw=3.0, alpha=0.9, label="BOLD signal")

        mean = np.mean(n_b_est_blocks, axis=1)
        std = np.std(n_b_est_blocks, axis=1)
        t = np.arange(n_b_est_blocks.shape[0]) * TR_HCP
        borders = (mean - std, mean + std)
        ax.fill_between(t, borders[0], borders[1], alpha=0.2, color='red')
        ax.plot(t, mean, color='red', lw=3.0, label="z")
        ax.plot(t, p_e, '--g', lw=3.0, label="PE")

        ax.tick_params(labelsize=30)
        ax.axhline(0.0, color='black')
        ax.set_xlim((0, N_SCANS_HCP*TR_HCP))
        ax.set_ylim((-0.7, 1.2))
        plt.xticks([0, int(DUR_RUN_HCP/2.0), int(DUR_RUN_HCP)], fontsize=20)
        plt.yticks([-.5, 0, 1], fontsize=20)
        ax.set_xlabel("Time [s]", fontsize=20)
        ax.set_ylabel("Signal change [%]", fontsize=20)
        plt.legend(fontsize=20, framealpha=0.3)

        plt.tight_layout()
        filename = os.path.join(lbda_dir, 'b_est_blocks_{0}.pdf'.format(trial))
        plt.savefig(filename, dpi=150)

        n_ref_hrfs = inf_norm(ref_hrfs[trial].T).T
        n_b_est_hrfs = inf_norm(b_est_hrfs[trial].T).T

        fig, ax = plt.subplots(figsize=(10, 5))

        mean = np.mean(n_ref_hrfs, axis=1)
        std = np.std(n_ref_hrfs, axis=1)
        t = np.arange(n_ref_hrfs.shape[0]) * TR_HCP
        borders = (mean - std, mean + std)
        ax.fill_between(t, borders[0], borders[1], alpha=0.2, color='blue')
        ax.plot(t, mean, color='blue', lw=3.0, alpha=0.9,
                label="Pedregosa (2015)")

        mean = np.mean(n_b_est_hrfs, axis=1)
        std = np.std(n_b_est_hrfs, axis=1)
        t = np.arange(n_b_est_hrfs.shape[0]) * TR_HCP
        borders = (mean - std, mean + std)
        ax.fill_between(t, borders[0], borders[1], alpha=0.2, color='red')
        ax.plot(t, mean, color='red', lw=3.0, label="blind deconvolution")
        ax.plot(canonical_hrf, '--g', lw=3.0, label="SPM HRF")

        ax.tick_params(labelsize=30)
        ax.axhline(0.0, color='black')
        ax.set_xlim((0, 24*TR_HCP))
        ax.set_ylim((-0.7, 1.2))
        plt.xticks([0, int(hrf_dur/2.0), int(hrf_dur)], fontsize=20)
        plt.yticks([-.5, 0, 1], fontsize=20)
        ax.set_xlabel("Time [s]", fontsize=20)
        ax.set_ylabel("Signal change [%]", fontsize=20)
        plt.legend(fontsize=20, framealpha=0.3)

        plt.tight_layout()
        filename = os.path.join(lbda_dir, 'b_est_hrfs_{0}.pdf'.format(trial))
        plt.savefig(filename, dpi=150)

        filename = os.path.join(lbda_dir, 'b_est_hrfs_{0}.pkl'.format(trial))
        with open(filename, 'wb') as pfile:
            pickle.dump(b_est_hrfs, pfile)

        filename = os.path.join(lbda_dir, 'b_est_blocks_{0}.pkl'.format(trial))
        with open(filename, 'wb') as pfile:
            pickle.dump(b_est_blocks, pfile)

print("All results save under '{0}'".format(root_dir))
