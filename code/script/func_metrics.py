import jax
import numpy as np
import brainpy.math as bm

# NumPy utilities for empirical and post hoc FC/FCD analysis.
def z_score(data, axis=1):
    return (data - data.mean(axis=axis, keepdims=True)) / data.std(axis=axis, keepdims=True)

def preprocess_time_series(data, downsample_rate, axis=1):
    data_prep = data[:, ::downsample_rate, :]
    return z_score(data_prep, axis=axis)

def compute_fc_np(signal):
    return np.nan_to_num(np.corrcoef(signal))

def compute_fcd_np(signal, window):
    [dim,time] = signal.shape
    mask_tril = ~np.tril(np.ones([dim,dim],dtype=bool))
    fc_timeline_vect_mat = np.zeros([(dim**2-dim)//2,time-window])
    for j in range(time-window):
        signal_tmp = signal[:,j:j+window]
        fc_timeline_vect = compute_fc_np(signal_tmp)[mask_tril]
        fc_timeline_vect_mat[:,j] = fc_timeline_vect

    fcd = np.corrcoef(fc_timeline_vect_mat.T)

    return fcd

# BrainPy/JAX utilities used inside differentiable training objectives.
def compute_FC(signal, ZSC=False):
    if ZSC:
        zscored_signal = z_score(signal, axis=1)
    else:
        zscored_signal = bm.normalize(signal, axis=1, epsilon=1e-6)

    FC = bm.matmul(zscored_signal.transpose(0,2,1), zscored_signal)
    return FC

def get_slices(window, step, length):
    slices = []
    for i in range(0, length - window + 1, step):
        slices.append(bm.arange(i, i + window))
    return slices

def compute_truncated_FC(signal, slice, ZSC=False):
    if ZSC:
        truncated_signal = z_score(signal[:, slice, :],axis=1)
    else:
        truncated_signal = bm.normalize(signal[:, slice, :], axis=1, epsilon=1e-6)
    return bm.matmul(truncated_signal.transpose(0,2,1), truncated_signal) / len(slice)

def compute_sliced_trFCD(signal, slices, ZSC=False):
    FCs = bm.stack(jax.tree_map(lambda s: compute_truncated_FC(signal, s, ZSC), slices)).transpose(1, 0, 2, 3) 
    triu_FC = bm.triu_indices(FCs.shape[3], 1) 
    FCs = FCs[:, :, triu_FC[0], triu_FC[1]] 
    FCs = FCs.transpose(0, 2, 1) 

    FCs = bm.normalize(FCs, axis=1)
    FCD = bm.matmul(FCs.transpose(0, 2, 1), FCs) / len(triu_FC[0])
    triu_FCD = bm.triu_indices(len(slices), 1)
    trFCD = FCD[:, triu_FCD[0], triu_FCD[1]]

    return trFCD

# Correlation similarity is used as a scale-insensitive FC objective.
def correlation_similarity(target, preds, axis=-1, epsilon=1e-6):
    target = bm.normalize(target, axis=axis, epsilon=epsilon)
    preds  = bm.normalize(preds,  axis=axis, epsilon=epsilon)
    return bm.mean(target*preds, axis=axis)

def correlation_similarity_flatt(x, y):
    return correlation_similarity(bm.flatten(x), bm.flatten(y))

correlation_similarity_batch = jax.vmap(correlation_similarity_flatt)

# Parameter sanitation guards long optimization runs against isolated NaNs.
def check_nan0d(Var, last_value, rng):
    val = Var.value
    if bm.isnan(val):
        Var.value = last_value + 1e-5*(rng.random() + 1)
    return Var

def check_nan(Var, last_value, rng):
    val = Var.value
    nan_idx = bm.isnan(val)
    if bm.any(nan_idx):
        val = val.at[nan_idx].set(last_value[nan_idx] + 1e-5*(rng.random(val[nan_idx].shape) + 1))
        Var.value = val

def check_neg(Var):
    val = Var.value
    neg_idx = val < 0
    if bm.any(neg_idx):
        val = val.at[neg_idx].set(0)
        Var.value = val

def check_para(rnnLayer, loss_fun_FC, data_dict, epoch_i, train_var_list, rng):
    
    if 'G' in train_var_list:
        check_nan0d(rnnLayer.G, data_dict['epoch_G'][epoch_i-1], rng)
    if 'w' in train_var_list:
        check_nan(rnnLayer.w, data_dict['epoch_w'][epoch_i-1], rng)
    if 'I' in train_var_list:
        check_nan(rnnLayer.I, data_dict['epoch_I'][epoch_i-1], rng)
    if 'SC' in train_var_list:
        check_nan(rnnLayer.struc_conn_matrix, data_dict['epoch_SC'][epoch_i-1], rng)
        # Non-negativity is enforced through the model's SC mapping when requested.
    if 'sigma' in train_var_list:
        check_nan(loss_fun_FC.sigma, data_dict['epoch_sigma'][epoch_i-1], rng)


    
