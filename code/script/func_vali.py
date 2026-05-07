import os
import sys
import jax
import time
import numpy as np
import pandas as pd

import brainpy as bp
import brainpy.math as bm

from func_metrics import *
from func_model import *
from func_settings import *

import warnings
warnings.filterwarnings('ignore')

# Validation simulates every saved training epoch and scores FC/FCD fit.
# The resulting CSV identifies the checkpoint used by the final test phase.

def validation(step, settings_list):
    bm.set_mode(bm.batching_mode)
    # Use deterministic seeds per validation step for reproducible trajectories.
    settings_dict = settings_list[step]
    rand_seed = settings_dict['random seed']
    Batch_size = settings_dict['batch size']
    window = int(settings_dict['window time']/settings_dict['TR'])
    bm.random.seed(int(rand_seed + step*1000))
    np.random.seed(int(rand_seed + step*1000))
    rng = bm.random.RandomState(int(rand_seed + step*1000))

    # Resolve output paths and empirical targets.
    species = settings_dict['species']
    atlas = settings_dict['atlas']
    save_dir, save_file = get_save_path(settings_dict)

    N, _, FC, biomarkers, _, _, _, _ = load_data(settings_dict)
    np.fill_diagonal(FC, 1.)
    tri_idx = np.triu_indices_from(FC, k=1)
    FC_vec = FC[tri_idx]

    # Empirical FCD distribution is stored as a CDF over off-diagonal entries.
    fcd_cdf_emp = biomarkers['fcd cdf']
    vmin = 0.
    vmax = 1.
    n_bins = 10000

    # Validation outputs are cached per training step to make interrupted runs resumable.
    ALL_COMPLETED = True
    for step_load in settings_dict['training steps']:
        load_dir, load_file = get_save_path(settings_dict=settings_list[step_load])
        load_path_vali = os.path.join(load_dir, load_file+'_vali.bp')
        if not os.path.exists(load_path_vali):
            ALL_COMPLETED = False
    
    if not ALL_COMPLETED:
        # Set simulation resolution and allocate reusable arrays.
        bm.dt = settings_dict['dt']
        duration = int(np.round(settings_dict['fMRI time']/bm.dt, 0))
        warmation = int(np.round(settings_dict['warm-up epoch long']/bm.dt, 0))
        downsample_rate = int(np.round(settings_dict['TR']/bm.dt, 0))

        # Allocate memory once and update values for each epoch.
        G = np.zeros((Batch_size, 1))
        w = np.zeros((Batch_size, N))
        I = np.zeros((Batch_size, N))
        sigma = np.zeros((Batch_size, 1, N))
        struc_conn_matrix = np.zeros((N, N))
        tr_noise_wm = np.zeros((Batch_size, warmation, N))
        tr_noise_run = np.zeros((Batch_size, duration, N))

        G = bm.asarray(G)
        w = bm.asarray(w)
        I = bm.asarray(I)
        sigma = bm.asarray(sigma)
        struc_conn_matrix = bm.asarray(struc_conn_matrix)
        tr_noise_wm = bm.asarray(tr_noise_wm)
        tr_noise_run = bm.asarray(tr_noise_run)

    # Containers for the summary CSV spanning all training steps.
    epoch_list = []
    epoch_ori_list = []
    corr_list = []
    mse_list = []
    ks_list = []
    step_list = []
    seed_list = []

    total_epoch_i = 0
    for step_load in settings_dict['training steps']:
        seed = settings_list[step_load]['random seed']
        epoch_N = settings_list[step_load]['epoch number']
        load_dir, load_file = get_save_path(settings_dict=settings_list[step_load])

        # Locate training, validation, and NaN-interrupted checkpoints.
        load_path_vali = os.path.join(load_dir, load_file+'_vali.bp')
        load_path_training = os.path.join(load_dir, load_file+'.bp')
        load_path_nan = os.path.join(load_dir, load_file+'_nan.bp')

        # Cache simulated BOLD, FC, and FCD arrays separately for inspection.
        save_bold_dir = os.path.join(save_dir, 'bold', save_file, 'step{}'.format(step_load))
        save_fc_dir = os.path.join(save_dir, 'fc', save_file, 'step{}'.format(step_load))
        save_fcd_dir = os.path.join(save_dir, 'fcd', save_file, 'step{}'.format(step_load))
        os.makedirs(save_bold_dir, exist_ok=True)
        os.makedirs(save_fc_dir, exist_ok=True)
        os.makedirs(save_fcd_dir, exist_ok=True)
        if not os.path.exists(load_path_vali):
            if not os.path.exists(load_path_training):
                # In automated workflows validation may start before training finishes.
                print('Training step {} of seed {} has not been completed'.format(step_load, rand_seed))
                wait_time = 0
                check_interval = 1
                while not os.path.exists(load_path_training):
                    if os.path.exists(load_path_nan):
                        raise ValueError('Training step {} of seed {} has been interrupted by NaN loss'.format(step_load, rand_seed))
                    time.sleep(check_interval)
                    wait_time += check_interval
                    print('\rHave been waiting for {} hours {} minutes {} seconds...'.format(wait_time//3600, (wait_time%3600)//60, wait_time%60), end='')

            # Load the complete optimization trajectory for epoch-wise validation.
            states = bp.checkpoints.load_pytree(load_path_training)
            epoch_FCcorr = np.zeros((epoch_N,))
            epoch_FCmse = np.zeros((epoch_N,))
            epoch_FCDks = np.zeros((epoch_N,))
            epoch_G = states['epoch_G']
            epoch_w = states['epoch_w']
            epoch_I = states['epoch_I']
            epoch_sigma = states['epoch_sigma']
            epoch_SC = states['epoch_SC']

            for epoch_i in range(epoch_N): # Epoch index within the training step.
                tic = time.time()
                print('#############Validation for {}-{} (seed={}-step{}-epoch={})###############'.format(species, atlas, rand_seed, step_load, epoch_i))
                # Per-epoch cached outputs.
                save_run_file = 'epoch={}.npy'.format(epoch_i)
                save_bold_path = os.path.join(save_bold_dir, save_run_file)
                save_fc_path = os.path.join(save_fc_dir, save_run_file)
                save_fcd_path = os.path.join(save_fcd_dir, save_run_file)

                # BOLD simulation.
                if not os.path.exists(save_bold_path):
                    # Load epoch parameters into reusable BrainPy arrays.
                    G.value = epoch_G[epoch_i].repeat(Batch_size, axis=0).reshape(Batch_size, 1)
                    w.value = np.expand_dims(epoch_w[epoch_i], axis=0).repeat(Batch_size, axis=0)
                    I.value = np.expand_dims(epoch_I[epoch_i], axis=0).repeat(Batch_size, axis=0)
                    sigma.value = np.expand_dims(epoch_sigma[epoch_i], axis=(0, 1)).repeat(Batch_size, axis=0)
                    struc_conn_matrix.value = bm.relu(epoch_SC[epoch_i])
                    
                    tr_noise_wm.value = rng.randn(Batch_size, warmation, N) * bm.sqrt(bm.dt) * bm.abs(sigma)
                    tr_noise_run.value = rng.randn(Batch_size, duration, N) * bm.sqrt(bm.dt) * bm.abs(sigma)

                    # Validation uses the Volterra readout for BOLD-like activity.
                    model = MFMVolterra(N, Batch_size, struc_conn_matrix, G, w, I, 
                                        TrainVar_list=settings_dict['training variables'],
                                        CST=settings_dict['contrain non-negative SC'],
                                        rng=rng)

                    # Warm-up followed by full-length resting-state simulation.
                    model.reset_state(Batch_size=Batch_size)
                    print('Warm-up model...')
                    runner_warmup = bp.DSTrainer(model, progress_bar=False)
                    bold_warmup = runner_warmup.predict(tr_noise_wm, reset_state=True)

                    print('Run model...')
                    runner_run = bp.DSTrainer(model, progress_bar=False)
                    bold_run = runner_run.predict(tr_noise_run, reset_state=False)

                    data_prep = preprocess_time_series(bold_run, downsample_rate)
                    np.save(save_bold_path, data_prep)
                else:
                    data_prep = np.load(save_bold_path)

                # Static FC.
                if not os.path.exists(save_fc_path):
                    FC_pred_batch = bm.matmul(data_prep.transpose(0,2,1), data_prep) / data_prep.shape[1]
                    np.save(save_fc_path, np.array(FC_pred_batch))
                else:
                    FC_pred_batch = np.load(save_fc_path)

                # FC correlation and MSE against empirical FC.
                FC_pred = FC_pred_batch.mean(axis=0)
                FC_pred_vec = FC_pred[tri_idx]
                corr_tmp = np.corrcoef(np.nan_to_num(FC_pred_vec), np.nan_to_num(FC_vec))[0,1]
                mse_tmp = np.mean((np.nan_to_num(FC_pred_vec) - np.nan_to_num(FC_vec))**2)

                # FCD distribution.
                if not os.path.exists(save_fcd_path):
                    fcd_list = []
                    fcd_entries_list = []
                    for batch_idx in range(Batch_size):
                        signal = data_prep[batch_idx].T
                        fcd = compute_fcd_np(signal, window)
                        fcd_list.append(fcd)
                        fcd_entries_list.extend(list(fcd[np.triu_indices(fcd.shape[0], k=1)]))
                    fcd_entries = np.array(fcd_entries_list)
                    np.save(save_fcd_path, np.array(fcd_list))
                else:
                    fcd_batch = np.load(save_fcd_path)
                    fcd_entries_list = []
                    for batch_idx in range(Batch_size):
                        fcd = fcd_batch[batch_idx]
                        fcd_entries_list.extend(list(fcd[np.triu_indices(fcd.shape[0], k=1)]))
                    fcd_entries = np.array(fcd_entries_list)

                # Kolmogorov-Smirnov distance between simulated and empirical FCD CDFs.
                fcd_hist_sim, fcd_bin_edges_sim = np.histogram(fcd_entries, bins=n_bins, range=(vmin, vmax))
                fcd_cumsum_sim = np.cumsum(fcd_hist_sim)
                fcd_cdf_sim = fcd_cumsum_sim / fcd_cumsum_sim[-1]
                ks_tmp = np.max(np.abs(fcd_cdf_sim - fcd_cdf_emp))

                # Append metrics for checkpoint selection.
                epoch_FCcorr[epoch_i] = corr_tmp
                epoch_FCmse[epoch_i] = mse_tmp
                epoch_FCDks[epoch_i] = ks_tmp

                print('FC Correlation: {:.3f}'.format(corr_tmp))
                print('FC MSE: {:.3f}'.format(mse_tmp))
                print('FCD KS Distance: {:.3f}'.format(ks_tmp))
                epoch_list.append(total_epoch_i)
                epoch_ori_list.append(epoch_i)
                corr_list.append(corr_tmp)
                mse_list.append(mse_tmp)
                ks_list.append(ks_tmp)
                step_list.append(step_load)
                seed_list.append(seed)
                total_epoch_i += 1
                toc = time.time()
                print('Time: {:.2f}s'.format(toc-tic))    

            # Save validation metrics beside the corresponding training checkpoint.
            states.update({'epoch_FCcorr_vali': epoch_FCcorr,
                           'epoch_FCmse_vali': epoch_FCmse,
                           'epoch_FCDks_vali': epoch_FCDks})
            bp.checkpoints.save_pytree(load_path_vali, states)
        else:
            states = bp.checkpoints.load_pytree(load_path_vali)
            epoch_FCcorr = states['epoch_FCcorr_vali']
            epoch_FCmse = states['epoch_FCmse_vali']
            epoch_FCDks = states['epoch_FCDks_vali']
            for epoch_i, corr_tmp in enumerate(epoch_FCcorr):
                epoch_list.append(total_epoch_i)
                epoch_ori_list.append(epoch_i)
                corr_list.append(corr_tmp)
                mse_list.append(epoch_FCmse[epoch_i])
                ks_list.append(epoch_FCDks[epoch_i])
                step_list.append(step_load)
                seed_list.append(seed)
                total_epoch_i += 1

        # Write the cross-step validation summary used by downstream analyses.
        df_vali = pd.DataFrame()
        df_vali['Epoch'] = epoch_list
        df_vali['FC correlation'] = corr_list
        df_vali['FC MSE'] = mse_list
        df_vali['FCD KS Distance'] = ks_list
        df_vali['Step'] = step_list
        df_vali['Seed'] = seed_list
        df_vali['Epoch (Original)'] = epoch_ori_list
        df_vali.to_csv(os.path.join(save_dir, save_file+'.csv'), index=False)
