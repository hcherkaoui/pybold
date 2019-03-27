# coding: utf-8
""" Set of usefull functions.
"""
import os
import csv
from datetime import datetime
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Memory, Parallel, delayed
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
from nistats.reporting import plot_design_matrix
from nistats.first_level_model import FirstLevelModel
from nistats.design_matrix import (make_first_level_design_matrix,
                                   check_design_matrix)
from pybold.bold_signal import hrf_estim
from hrf_estimation.hrf import spmt, dspmt, ddspmt
from hrf_estimation.rank_one_ import glm
from hrf_estimation.savitzky_golay import savgol_filter
from hcp_builder.dataset import get_data_dirs, download_experiment


EPOCH_DUR_HCP = 12.0
DUR_RUN_HCP = 3*60 + 34
N_SCANS_HCP = 284
TR_HCP = DUR_RUN_HCP / float(N_SCANS_HCP)


def create_result_dir(sufixe='', root_path='.'):
    """Return the path of a result directory."""
    d = datetime.now()
    dirname = 'results_{0}#{1}{2}{3}{4}'.format(sufixe, d.day, d.hour,
                                                d.minute, d.second)
    dirname = os.path.join(root_path, dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def mask_n_max(a, n):
    """Return the mask (same array of a) to get the n maximums of a"""
    indices = np.argpartition(a.flat, kth=-n)
    mask = np.zeros_like(a, dtype=bool)
    mask.flat[indices[-n:]] = True
    return mask


def get_hcp_fmri_fname(subject_id, anat_data=False):
    """Return the tfMRI filename."""
    data_dir = get_data_dirs()[0]
    path = os.path.join(data_dir, str(subject_id))
    if not os.path.exists(path):
        download_experiment(subject=subject_id, data_dir=None,
                            data_type='task', tasks='MOTOR', sessions=None,
                            overwrite=True, mock=False, verbose=10)
    fmri_path = path
    fmri_dirs = ['MNINonLinear', 'Results', 'tfMRI_MOTOR_RL',
                 'tfMRI_MOTOR_RL.nii.gz']
    for dir_ in fmri_dirs:
        fmri_path = os.path.join(fmri_path, dir_)
    anat_path = path
    anat_dirs = ['MNINonLinear', 'Results', 'tfMRI_MOTOR_RL',
                 'tfMRI_MOTOR_RL_SBRef.nii.gz']
    for dir_ in anat_dirs:
        anat_path = os.path.join(anat_path, dir_)
    if anat_data:
        return fmri_path, anat_path
    else:
        return fmri_path


def get_paradigm_hcp(subject_id, data_dir=None):
    """Return onsets, conditions of the HCP task experimental protocol."""
    data_dir = get_data_dirs()[0]
    path = os.path.join(data_dir, str(subject_id))
    dirs = ['MNINonLinear', 'Results', 'tfMRI_MOTOR_RL', 'EVs']
    for directory in dirs:
        path = os.path.join(path, directory)
    l_files = glob(os.path.join(path, "*"))
    l_files = [f for f in l_files if 'Sync.txt' not in f]

    tmp_dict = {}
    for filename in l_files:
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            cond = os.path.splitext(os.path.basename(filename))[0]
            tmp_dict[cond] = [row for row in reader]

    onsets = []
    for trial_type, raw_onsets in tmp_dict.iteritems():
        for t in raw_onsets:
            t[-1] = trial_type
            onsets.append(t)

    df = pd.DataFrame(onsets, columns=['onset', 'duration', 'trial_type'])
    serie_ = pd.Series(np.zeros(df.shape[0], dtype=int), index=df.index)
    df.insert(0, 'session', serie_)
    tmp = df[['onset', 'duration']].apply(pd.to_numeric)
    df[['onset', 'duration']] = tmp
    df = df.sort_values('onset')
    df = df[['session', 'trial_type', 'onset', 'duration']]
    df.reset_index(inplace=True, drop=True)

    return df, np.linspace(0.0, DUR_RUN_HCP, N_SCANS_HCP)


def get_protocol_hcp(subject_id, trial, data_dir=None):
    """Get the HCP motor task protocol."""
    paradigm_full, _ = get_paradigm_hcp(subject_id)
    paradigm = paradigm_full[paradigm_full['trial_type'] == trial]

    onset = []
    for t, d in zip(paradigm['onset'], paradigm['duration']):
        onset.extend(list(t + np.arange(int(d / TR_HCP)) * TR_HCP))
    trial_type = [trial] * len(onset)
    onset = np.array(onset)
    trial_type = np.array(trial_type)

    t = np.linspace(0.0, DUR_RUN_HCP, N_SCANS_HCP)

    p_e = np.zeros(N_SCANS_HCP)
    mask_cond = (paradigm['trial_type'] == trial).values
    onset_t = paradigm['onset'][mask_cond].values
    onset_idx = (onset_t / TR_HCP).astype(int)
    durations = paradigm['duration'][mask_cond].values
    for i, d in zip(onset_idx, durations):
        p_e[i:int(i+d)] = 1.0

    return trial_type, onset, p_e, t


def hrf_coef_to_hrf(hrfs_coef, dur, t_r):
    """Return the HRFs from the given coefs."""
    t_hrf = np.linspace(0.0, dur, int(dur/t_r))
    hrfs = hrfs_coef[0] * spmt(t_hrf)[:, None]
    hrfs += hrfs_coef[1] * dspmt(t_hrf)[:, None]
    hrfs += hrfs_coef[2] * ddspmt(t_hrf)[:, None]
    return hrfs


def _hcp_regions_selection(fmri_img, subject_id, n_voxels=10, n_jobs=1,
                           verbose=False):
    """GLM on HCP dataset."""
    paradigm, t_frames = get_paradigm_hcp(subject_id)

    d_m = make_first_level_design_matrix(
                             t_frames, paradigm, hrf_model='spm',
                             drift_model='Cosine',
                             period_cut=2 * 2 * EPOCH_DUR_HCP)

    glm = FirstLevelModel(t_r=TR_HCP, slice_time_ref=0.0, noise_model='ols',
                          min_onset=10.0, signal_scaling=False,
                          smoothing_fwhm=6.0, standardize=False,
                          memory_level=1, memory='./.cachedir',
                          minimize_memory=False, n_jobs=n_jobs)

    glm.fit(run_imgs=fmri_img, design_matrices=d_m)

    _, _, names = check_design_matrix(d_m)
    n_names = len(names)
    c_val = dict([(n, c) for n, c in zip(names, np.eye(n_names))])
    c_val['rh-lh'] = c_val['rh'] - c_val['lh']
    c_val['lh-rh'] = c_val['lh'] - c_val['rh']
    z_maps = dict([(n, glm.compute_contrast(c, output_type='z_score'))
                   for n, c in c_val.iteritems()])

    z_maps = {'rh': z_maps['rh-lh'], 'lh': z_maps['lh-rh'],
              'cue': z_maps['cue']}
    region_mask_imgs = {}
    for name, _ in [('rh', 'rh-lh'), ('lh', 'lh-rh'), ('cue', 'cue')]:
        z_map_vector_mask = glm.masker_.transform(z_maps[name]).flatten()
        z_region_vector_mask = mask_n_max(z_map_vector_mask, n_voxels)
        z_region_vector_mask = z_region_vector_mask.astype(float)
        region_mask_imgs[name] = \
                            glm.masker_.inverse_transform(z_region_vector_mask)

    return d_m, z_maps, region_mask_imgs


hcp_regions_selection_cached = \
                           Memory('./.cachedir').cache(_hcp_regions_selection)


def _hcp_hrf_comparison(fmri_img, subject_id, n_voxels=10, hrf_dur=30.0,
                        n_jobs=1, verbose=False):
    """Return Cherkaoui and Pedregosa estimated HRF for comparison."""
    _, _, region_masks = hcp_regions_selection_cached(fmri_img, subject_id,
                                                      n_voxels=n_voxels,
                                                      n_jobs=n_jobs)

    est_hrfs, ref_hrfs, voxels = {}, {}, {}
    for trial, region_mask in region_masks.iteritems():
        trial_type, onset, p_e, _ = get_protocol_hcp(subject_id, trial)
        masker = NiftiMasker(t_r=TR_HCP, standardize=True,
                             mask_img=region_mask)
        voxels[trial] = masker.fit_transform(fmri_img)
        voxels[trial] -= savgol_filter(voxels[trial], 91, 3, axis=0)

        res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(hrf_estim)(
                                 p_e, voxel, t_r=TR_HCP,
                                 dur=hrf_dur, bounds=True, verbose=0)
                                 for voxel in voxels[trial].T)
        est_hrfs[trial] = np.vstack([hrf for hrf, J in res]).T

        hrfs_coef, _ = glm(trial_type, onset, TR_HCP, voxels[trial],
                           drifts=np.zeros((voxels[trial].shape[0], 1)),
                           mode='r1glm', basis='3hrf',
                           n_jobs=n_jobs, verbose=0,
                           return_design_matrix=False)
        ref_hrfs[trial] = hrf_coef_to_hrf(hrfs_coef, hrf_dur, TR_HCP)

    return est_hrfs, ref_hrfs, voxels


hcp_hrf_comparison_cached = Memory('./.cachedir').cache(_hcp_hrf_comparison)


def plot_trial_z_map(z_map, title='Z map', th=4.0, bg=None, save=False,
                     display_mode='ortho', filename='z_map.png',
                     verbose=False):
    """Plot z-map w.r.t given trial."""
    disp = plot_stat_map(z_map, title=title, threshold=th, bg_img=bg,
                         display_mode=display_mode)
    if save:
        if verbose:
            print("Saving plot under '{0}'".format(filename))
        disp.savefig(filename)
    else:
        plt.show()


def plot_d_m(dm, save=False, filename='dm.png', dir_name='.', verbose=False):
    """Plot designa matrix."""
    plot_design_matrix(dm)
    if save:
        filename = os.path.join(dir_name, filename)
        if verbose:
            print("Saving plot under '{0}'".format(filename))
        plt.savefig(filename)
    else:
        plt.show()
