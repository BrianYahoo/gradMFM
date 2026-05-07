def settings_humanGlasser():
    # Stage 0 estimates global and regional neural parameters with a linear readout.
    set0 = {'step name': 'pretrain',
            'warm-up epoch long': 10,
            'simulation epoch long': 60,
            'batch size': 100,
            'learning rate': 1e-2,
            'epoch number': 1000,
            'training variables': ['G', 'w', 'I', 'sigma'],
            'outLayer': 'linear',
            'more accurate z-score': False,
            'loss': ['fc']}
    # Stage 1 starts structural-connectome inference while retaining the linear readout.
    set1 = {'step name': 'train-conn-ac',
            'warm-up epoch long': 10,
            'simulation epoch long': 60,
            'batch size': 100,
            'learning rate': 1e-3,
            'epoch number': 500,
            'training variables': ['SC', 'I', 'sigma'],
            'outLayer': 'linear',
            'more accurate z-score': False,
            'loss': ['fc']}
    # Stage 2 refines the inferred connectome with a Volterra hemodynamic readout.
    set2 = {'step name': 'train-conn-fc',
            'warm-up epoch long': 30,
            'simulation epoch long': 3*60,
            'batch size': 25,
            'learning rate': 1e-4,
            'epoch number': 500,
            'training variables': ['SC', 'I', 'sigma'],
            'outLayer': 'Volterra',
            'more accurate z-score': True,
            'loss': ['fc']}
    # Stage 3 jointly fits static FC and dynamic FCD biomarkers.
    set3 = {'step name': 'train-conn-fcd',
            'warm-up epoch long': 30,
            'simulation epoch long': 3*60,
            'batch size': 15,
            'learning rate': 1e-5,
            'epoch number': 500,
            'training variables': ['SC', 'I', 'sigma'],
            'outLayer': 'Volterra',
            'more accurate z-score': True,
            'loss': ['fc', 'fcd']}
    training_dicts_list = [set0, set1, set2, set3]
    return training_dicts_list
