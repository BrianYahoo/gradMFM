import os
import jax
import time
import numpy as np

import brainpy as bp
import brainpy.math as bm

from func_metrics import *
from func_model import *
from func_loss import *
from func_settings import *

import warnings

def training(step, settings_list):
    bm.set_mode(bm.training_mode)

    settings_dict = settings_list[step]
    training_steps = settings_dict['training steps']
    if step != training_steps[0]:
        settings4load_dict = settings_list[step-1]
    else:
        settings4load_dict = None

    rand_seed = settings_dict['random seed']
    Batch_size = settings_dict['batch size']
    epoch_N = settings_dict['epoch number']
    bm.random.seed(int(rand_seed + step*1000))
    np.random.seed(int(rand_seed + step*1000))
    rng = bm.random.RandomState(int(rand_seed + step*1000))

    save_dir, save_file = get_save_path(settings_dict)
    save_path_train = os.path.join(save_dir, save_file+'.bp')
    if os.path.exists(save_path_train):
        print('Training step {} ({}) of seed {} has been completed!'.format(step, settings_dict['step name'], rand_seed))
        return

    # load data
    N, struc_conn_matrix, FC, biomarkers, G, w, I, sigma = load_data(settings_dict, settings4load_dict)
    ###########################################################################################################
    # Set up the model
    bm.dt = settings_dict['dt']
    duration = int(np.round(settings_dict['simulation epoch long']/bm.dt, 0))
    warmation = int(np.round(settings_dict['warm-up epoch long']/bm.dt, 0))
    struc_conn_matrix = bm.asarray(struc_conn_matrix).cuda()
    
    if settings_dict['outLayer'] == 'linear':
        model = MFM(N, Batch_size, struc_conn_matrix, G, w, I, 
                    TrainVar_list=settings_dict['training variables'], 
                    CST=settings_dict['contrain non-negative SC'],
                    rng=rng)
    elif settings_dict['outLayer'] == 'Volterra':
        model = MFMVolterra(N, Batch_size, struc_conn_matrix, G, w, I, 
                            TrainVar_list=settings_dict['training variables'],
                            CST=settings_dict['contrain non-negative SC'],
                            rng=rng)
    model.reset_state(Batch_size=Batch_size)

    ###########################################################################################################
    # random noise inputs
    tr_noise_wm = rng.randn(Batch_size, warmation, N) * bm.sqrt(bm.dt) * bm.abs(sigma)
    tr_noiseinput = rng.randn(Batch_size, duration, N) * bm.sqrt(bm.dt) 

    tr_FC = np.expand_dims(FC,0).repeat(Batch_size,axis=0)
    mask=np.ones_like(FC)
    tr_mask_FC = np.expand_dims(mask,0).repeat(Batch_size,axis=0) / duration # for Cov calculation (1/n)X*XT

    if 'fcd' in settings_dict['loss']:
        trFCDmean = biomarkers['fcd mean (time={} window={} step={})'.format(
            int(settings_dict['simulation epoch long']), 
            int(settings_dict['window time']), 
            int(settings_dict['window step']))]
        trFCDstd = biomarkers['fcd std (time={} window={} step={})'.format(
            int(settings_dict['simulation epoch long']), 
            int(settings_dict['window time']), 
            int(settings_dict['window step']))]
        slices = get_slices(window=int(np.round(settings_dict['window time'] / bm.dt, 0)),
                            step=int(np.round(settings_dict['window step'] / bm.dt, 0)), 
                            length=duration)
        trFCDmean = bm.asarray(trFCDmean).cuda()
        trFCDstd = bm.asarray(trFCDstd).cuda()

    # move data to GPU
    tr_noise_wm = bm.asarray(tr_noise_wm).cuda()
    tr_noiseinput = bm.asarray(tr_noiseinput).cuda()
    tr_FC = bm.asarray(tr_FC).cuda()
    tr_mask_FC = bm.asarray(tr_mask_FC).cuda()
    ###########################################################################################################
    if settings_dict['loss'] == ['fc']:
        loss_func = loss_function_fc(model, N, sigma, 
                                     TrainVar_list=settings_dict['training variables'], 
                                     ZSC=settings_dict['more accurate z-score'])
        tr_tuple = (tr_FC, tr_mask_FC)
    elif settings_dict['loss'] == ['fc', 'fcd']:
        loss_func = loss_function_fcd(model, N, sigma, 
                                      TrainVar_list=settings_dict['training variables'], 
                                      ZSC=settings_dict['more accurate z-score'],)
        tr_tuple = (tr_FC, tr_mask_FC, trFCDmean, trFCDstd, slices)

    train_dict = model.train_vars().unique()
    train_dict.update(loss_func.train_vars().unique())
    grad_fun_FC = bm.grad(loss_func, grad_vars=train_dict, has_aux=True, return_value=True)
    opt_FC = bp.optim.Adam(lr=settings_dict['learning rate'],
                           train_vars=train_dict)

    # training function
    @bm.jit
    def train_FC(inputs, tr_tuple):
        grads, loss, aux_metrics = grad_fun_FC(inputs, tr_tuple)
        grads = jax.tree_util.tree_map(lambda g:bm.clip_by_value(g,-1e5,1e5), grads) # clipping gradient by a big value,防止梯度爆炸
        opt_FC.update(grads)
        return grads, loss, aux_metrics

    ###########################################################################################################
    # training
    data_dict = {'epoch_loss': np.zeros((epoch_N,)), 
                    'epoch_G': np.zeros((epoch_N,)),
                    'epoch_w': np.zeros((epoch_N,N)),
                    'epoch_I': np.zeros((epoch_N,N)),
                    'epoch_sigma': np.zeros((epoch_N,N)),
                    'epoch_SC': np.zeros((epoch_N,N,N)),}
    if 'fc' in settings_dict['loss']:
        data_dict.update({'epoch_FCcor': np.zeros((epoch_N,)),
                          'epoch_FCmse': np.zeros((epoch_N,)),})
    if 'fcd' in settings_dict['loss']:
        data_dict.update({'epoch_FCDmean': np.zeros((epoch_N,)),
                          'epoch_FCDstd': np.zeros((epoch_N,)),})

    for epoch_i in range(epoch_N): # epoch number
        tic = time.time()

        # warm-up model
        print('#############Training seed={}-step{}-epoch={}###############'.format(rand_seed, step, epoch_i))
        print('Warm-up model...')
        tr_noise_wm.value = rng.randn(Batch_size, warmation, N) * bm.sqrt(bm.dt) * bm.abs(sigma)
        model.reset_init()
        runner = bp.DSTrainer(model, progress_bar=False, numpy_mon_after_run=False,)
        warmup_output = runner.predict(tr_noise_wm, reset_state=True)

        # get mini-batch
        tr_noiseinput.value = rng.randn(Batch_size, duration, N) * bm.sqrt(bm.dt) 
        
        # training
        print('Training model...')
        grads, loss, metric_tuple = train_FC(tr_noiseinput, tr_tuple)

        if bm.isnan(loss):
            nan_path = os.path.join(save_dir, save_file+'_nan.bp')
            states = {
                'model': model.state_dict(), 
                'loss_func': loss_func.state_dict(), 
                'optimizerFC': opt_FC.state_dict()
                }
            states.update(data_dict)
            states.update(settings_dict)
            bp.checkpoints.save_pytree(nan_path, states)
            raise ValueError('Loss is NaN for seed={}-step{}-epoch={}'.format(rand_seed, step, epoch_i))
            # raise warning
            # warnings.warn('Loss is NaN for seed={}-step{}-epoch={}'.format(rand_seed, step, epoch_i))
            # return
        
        # printing results
        print('Loss: {:.3f}'.format(loss))
        if settings_dict['loss'] == ['fc']:
            FCmse, FCcor = metric_tuple
            print('FC mse: {:.3f}'.format(FCmse))
            print('FC correlation: {:.3f}'.format(FCcor))
        elif settings_dict['loss'] == ['fc', 'fcd']:
            FCmse, FCcor, FCDmean, FCDstd = metric_tuple
            print('FC mse: {:.3f}'.format(FCmse))
            print('FC correlation: {:.3f}'.format(FCcor))
            print('FCD mean difference: {:.3f}'.format(FCDmean))
            print('FCD std difference: {:.3f}'.format(FCDstd))

        # appending results
        data_dict['epoch_loss'][epoch_i] = loss
        
        if 'fc' in settings_dict['loss']:
            data_dict['epoch_FCcor'][epoch_i] = FCcor
            data_dict['epoch_FCmse'][epoch_i] = FCmse
        if 'fcd' in settings_dict['loss']:
            data_dict['epoch_FCDmean'][epoch_i] = FCDmean
            data_dict['epoch_FCDstd'][epoch_i] = FCDstd
        
        data_dict['epoch_G'][epoch_i] = np.asarray(model.rnnLayer.G)
        data_dict['epoch_w'][epoch_i, :] = np.asarray(model.rnnLayer.w)
        data_dict['epoch_I'][epoch_i, :] = np.asarray(model.rnnLayer.I)
        data_dict['epoch_sigma'][epoch_i, :] = np.asarray(bm.abs(loss_func.sigma))
        data_dict['epoch_SC'][epoch_i, :, :] = np.asarray(bm.relu(model.rnnLayer.struc_conn_matrix))

        # checking parameters
        # print('Checking parameters...')
        check_para(model.rnnLayer, loss_func, data_dict, epoch_i, settings_dict['training variables'], rng)

        toc = time.time()
        print('Time: {:.2f}s'.format(toc-tic))

    ###########################################################################################################
    # save
    states = {'model': model.state_dict(), 
                'loss_func': loss_func.state_dict(), 
                'optimizerFC': opt_FC.state_dict()}
    states.update(data_dict)
    states.update(settings_dict)
    bp.checkpoints.save_pytree(save_path_train, states)