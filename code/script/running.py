import os
import sys

import brainpy as bp
import brainpy.math as bm

from func_settings import get_settings_list
from func_train import training
from func_vali import validation
from func_test import test
        
def running(step, settings_list):
    if step in settings_list[0]['training steps']:
        training(step, settings_list)
    elif step in settings_list[0]['validation steps']:
        validation(step, settings_list)
    elif step in settings_list[0]['test steps']:
        test(step, settings_list)

if __name__ == '__main__':
    gpu_id = str(sys.argv[1])
    species = str(sys.argv[2])
    atlas = str(sys.argv[3])
    metric = str(sys.argv[4])
    approach = str(sys.argv[5])
    seed = int(sys.argv[6])
    step = int(sys.argv[7])

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if int(gpu_id) <= -1:
        print('No GPU is used.')
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        bm.set_platform('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
        bm.set_platform('gpu')
        bm.gpu_memory_preallocation(percent=0.99)

    settings_list = get_settings_list(seed, species, atlas, metric, approach,)
    running(step, settings_list)