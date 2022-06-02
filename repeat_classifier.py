#####################################################################################################################################################
#####################################################################################################################################################
"""
Repeat the same experiments several times with differend fixed seeds.
"""
#####################################################################################################################################################
#####################################################################################################################################################

import toml
import torch
import random 
import numpy as np

from train_classifier import train
from copy import deepcopy

#####################################################################################################################################################
#####################################################################################################################################################

if __name__ == '__main__':

    # load the config file
    config = toml.load("config/pretrained_classifier.toml")

    # seeds = list(range(0,100))
    seeds = [0,1,2,3,4,5,6,7,8,9]
    # seeds = [0,1,2,3]
    # seeds = [4,5,6]
    # seeds = [7,8,9]

    # for each of the number of epochs
    for seed in seeds:

        # make a copy of the loaded config
        current_config = deepcopy(config)

        # reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        # train the model with the updated config
        train(current_config)

#####################################################################################################################################################