'''
Functions related to the device (GPU)
'''
import logging
import torch
from torch.cuda import get_device_name


def get_device(verbose=True) -> torch.device:
    '''
    Gets pytorch current available device
    '''
    if torch.cuda.is_available() is True:
        device = torch.device("cuda:0")
        logging.info("CUDA device found: " + str(device))
        logging.info("GPU: " + str(get_device_name(0)))
        GPU_MEMORY_GBS = torch.cuda.get_device_properties(device).total_memory / 1000000000
        logging.info("TOTAL GPU MEMORY: {}".format(GPU_MEMORY_GBS))
    else:
        device = torch.device("cpu")
        logging.info("CUDA device not available. Using CPU.")
    return device
