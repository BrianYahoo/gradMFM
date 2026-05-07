import os
import numpy as np

import brainpy as bp
import brainpy.math as bm

from set_hmG import settings_humanGlasser

# Experiment configuration and data-loading utilities.
# Paths are relative to code/bash or code/script, matching the release workflows.

def get_general_dict(seed,
                    species, atlas, 
                    metric, approach,):
    # Parameters shared across training, validation, and test phases.
    general_dict = {
        'species': species,
        'atlas': atlas,
        'metric': metric,
        'approach': approach,
        'initial G': 1.0,
        'initial w': 0.9,
        'initial I': 0.3,
        'initial sigma': 1e-3,
        'dt': 1e-3,
        'window time': 60, 
        'window step': 10,
        'random seed': seed,
        'contrain non-negative SC': True,
        }
    return general_dict

def get_species_dict(species, atlas):
    # Dataset-specific acquisition and biomarker settings.
    if species == 'human':
        if atlas == 'Glasser':
            species_dict = {
                'fMRI time': 14.4*60,
                'TR': 0.72,
                'lambda': 6.99,
                }
    return species_dict

def training_dict_list(species, atlas):
    # Select the training schedule for the requested species and atlas.
    if species == 'human':
        if atlas == 'Glasser':
            return settings_humanGlasser()
    
def validation_dict_list(species, atlas):
    # Validation simulates each saved training epoch without updating parameters.
    vali_dict = {
        'step name': 'validation',
        'warm-up epoch long': 2*60,
        'batch size': 1,
        'training variables': [],
    }
    return [vali_dict]

def test_dict_list(species, atlas):
    # The test phase evaluates the validation-selected epoch in larger batches.
    test_dict = {
        'step name': 'test',
        'warm-up epoch long': 2*60,
        'batch size': 30,
        'training variables': [],
    }
    return [test_dict]

def get_settings_list(seed, species, atlas, metric, approach):
    # Merge phase-specific dictionaries with shared metadata and step indices.
    general_dict = get_general_dict(seed, species, atlas, metric, approach)
    species_dict = get_species_dict(species, atlas)
    training_dicts_list = training_dict_list(species, atlas)
    validation_dicts_list = validation_dict_list(species, atlas)
    test_dicts_list = test_dict_list(species, atlas)

    training_steps = list(range(len(training_dicts_list)))
    validation_steps = list(range(training_steps[-1]+1, training_steps[-1]+1+len(validation_dicts_list)))
    test_steps = list(range(validation_steps[-1]+1, validation_steps[-1]+1+len(test_dicts_list)))
    all_steps = training_steps + validation_steps + test_steps
    all_settings_list = training_dicts_list + validation_dicts_list + test_dicts_list
    
    step_dict = {'training steps': training_steps,
                 'validation steps': validation_steps,
                 'test steps': test_steps,
                 'all steps': all_steps}
    for step in all_steps:
        all_settings_list[step].update({'step': step})
        all_settings_list[step].update(general_dict)
        all_settings_list[step].update(species_dict)
        all_settings_list[step].update(step_dict)

    return all_settings_list

def get_save_path(settings_dict, main_dir='../../data/results/'):
    # Standard checkpoint layout: species/atlas/step_seed.
    save_dir0 = '{}/'.format(settings_dict['species'])
    save_dir1 = '{}/'.format(settings_dict['atlas'])
    save_dir2 = 'step{}_{}/'.format(settings_dict['step'], settings_dict['step name'])
    save_dir = os.path.join(main_dir, save_dir0, save_dir1, save_dir2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = 'seed={}'.format(settings_dict['random seed'])

    return save_dir, save_file

def get_fig_dir(settings_dict):
    # Figures mirror the result hierarchy under figures/results.
    save_dir0, save_dir1 = get_save_path(settings_dict, main_dir='../../figures/results/')
    save_dir = os.path.join(save_dir0, save_dir1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir

def load_data(settings_dict, settings4load_dict=None):
    # The first training step initializes from empirical SC; later training steps
    # resume from the previous checkpoint. Validation/test use saved parameters.
    step = settings_dict['step']
    training_steps = settings_dict['training steps']
    if step == training_steps[0]:
        sc_load = np.load('../../data/input/{}/{}/{}_{}.npy'.format(settings_dict['species'], 
                                                                   settings_dict['atlas'], 
                                                                   settings_dict['metric'], 
                                                                   settings_dict['approach']))
        G = settings_dict['initial G']
        w = settings_dict['initial w']
        I = settings_dict['initial I']
        sigma = settings_dict['initial sigma']
        SC = sc_load / sc_load.max()

        print('SC min:', SC.min())
        print('SC min (positive):', SC[SC>0].min())
        print('SC max:', SC.max())

    elif step in training_steps and settings4load_dict is not None:
        # Continue the staged optimization from the final epoch of the previous step.
        load_dir, load_file = get_save_path(settings_dict=settings4load_dict)
        states = bp.checkpoints.load_pytree(os.path.join(load_dir, load_file+'.bp'))
        best_idx = -1

        G = states['epoch_G'][best_idx]
        w = states['epoch_w'][best_idx, :]
        I = states['epoch_I'][best_idx, :]
        sigma = states['epoch_sigma'][best_idx, :]
        SC = states['epoch_SC'][best_idx, :, :]

        print('SC min:', SC.min())
        print('SC min (positive):', SC[SC>0].min())
        print('SC max:', SC.max())
    
    else:
        G = None
        w = None
        I = None
        sigma = None
        SC = None

    # Empirical FC and precomputed FCD biomarkers are shared across phases.
    FC = np.load('../../data/input/{}/{}/fc.npy'.format(settings_dict['species'], 
                                                         settings_dict['atlas']))
    np.fill_diagonal(FC, 1.)
    n_roi = FC.shape[0]

    biomarkers = np.load('../../data/input/{}/{}/biomarkers.npz'.format(settings_dict['species'], 
                                                                         settings_dict['atlas']))

    return n_roi, SC, FC, biomarkers, G, w, I, sigma
