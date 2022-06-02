##############################################################################################################################################################
##############################################################################################################################################################
"""
Evaluate an ensemble of classification models after training.

Simply insert the experiments to be evaluated inside the following list.
The folders need to be located inside the results folder.

This script will create a new experiment folder containing the results.

# list of all experiments to check
experiments = [


]
"""
##############################################################################################################################################################
##############################################################################################################################################################

import os
import sys
import toml
import torch
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("pdf")
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif', serif='Times New Roman')
plt.rc('axes', labelsize=8)
plt.rc('font', size=8)  
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('savefig', dpi=300) 

import utils
import eval_cae

##############################################################################################################################################################
##############################################################################################################################################################

def ood_ensemble(experiments_models, train_config, config, folders, folder_ensemble, which_split, which_dataset=None):

    print("Evaluation using ensemble ...")

    if which_dataset is None: 
        which_dataset = which_split
    else:
        which_dataset = which_dataset + "_" + which_split

    # get the evaluation data
    ood_loader = eval_cae.get_setup(config, folders, which_split=which_split, nbr_of_samples_per_class=eval_cae.SAMPLES_PER_CLASS_FOR_OOD[config["dataset"]["name"]], load_model=False, print_dataset=False)

    if train_config["dataset"]["name"].lower() == "sviro_uncertainty":
        if "seats" in train_config["dataset"]["split"].lower():
            test_loader = eval_cae.get_setup(train_config, folders, which_split="test-adults-and-seats", nbr_of_samples_per_class=eval_cae.SAMPLES_PER_CLASS_FOR_OOD[train_config["dataset"]["name"]], load_model=False, print_dataset=False)
        else:
            test_loader = eval_cae.get_setup(train_config, folders, which_split="test-adults", nbr_of_samples_per_class=eval_cae.SAMPLES_PER_CLASS_FOR_OOD[train_config["dataset"]["name"]], load_model=False, print_dataset=False)
    else:
        test_loader = eval_cae.get_setup(train_config, folders, which_split="test", nbr_of_samples_per_class=eval_cae.SAMPLES_PER_CLASS_FOR_OOD[train_config["dataset"]["name"]], load_model=False, print_dataset=False)


    # keep track
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    true_negative = defaultdict(int)
    false_negative = defaultdict(int)
    ood_entropy = []
    eval_entropy = []

    # certainty thresholds to consider
    threshold_to_consider = np.linspace(start=0.0, stop=1.0, num=50, endpoint=False) 

    # make sure we are in eval mode
    for current_model in experiments_models.values():
        current_model.eval()

    # do not keep track of gradients
    with torch.no_grad():

        # for each batch of training images
        for idx_data_loader, data_loader in enumerate([test_loader, ood_loader]):

            # for each batch of training images
            for batch in data_loader:

                # push to gpu
                input_images = batch["image"].to(config["device"])

                # keep track of several iterations
                keep_track_predictions = []

                for current_model in experiments_models.values():

                    # encode the images
                    prediction = current_model(input_images)["prediction"].cpu()

                    # make them  one hot encodeed vectors
                    softmax_prediction = torch.nn.functional.softmax(prediction, dim=1)

                    # keep track
                    keep_track_predictions.append(softmax_prediction.numpy())

                # make sure its a numpy array of the correct shape
                # (nbr_trials, batch_size, nbr_classes)
                keep_track_predictions = np.stack(keep_track_predictions)
                # (batch_size, nbr_trials, nbr_classes)
                keep_track_predictions = np.transpose(keep_track_predictions, [1,0,2])

                # mean over the number of trials to get for each batch element a list
                # mean probability vector of values between 0 and 1
                class_probability = keep_track_predictions.mean(axis=1)

                # get the class with most counts as prediction
                prediction = class_probability.argmax(axis=1)
                prediction = np.array([int(x) for x in prediction])

                # entropy, normalize by log(nbr_classes) to get values between 0 and 1
                # log is the natural logarithm as used by stats.entropy as well
                uncertainty = stats.entropy(class_probability, axis=1) / np.log(current_model.nbr_classes)

                if idx_data_loader == 0:
                    eval_entropy.extend(uncertainty)
                else:
                    ood_entropy.extend(uncertainty)


                # for each threshold
                for threshold in threshold_to_consider:

                    # get only the predictions for which the probability is high enough
                    certain_mask = uncertainty < threshold
                    uncertain_mask = uncertainty >= threshold

                    if idx_data_loader == 0:
                        true_positive[threshold] += certain_mask.sum()
                        false_negative[threshold] += uncertain_mask.sum()
                    else:
                        true_negative[threshold] += uncertain_mask.sum()
                        false_positive[threshold] += certain_mask.sum()

    utils.compute_ood_scores(
        true_positive, 
        true_negative, 
        false_positive, 
        false_negative, 
        threshold_to_consider, 
        which_dataset, 
        {"data-ood":folder_ensemble},
        idx_recursion=-1,
        which_classifier=None,
        fix_dropout_mask=None,
    )

    entropy_folder = folder_ensemble.parent / "entropy"
    entropy_folder.mkdir(exist_ok=True)

    filepath_ood = entropy_folder / f"OOD_train_{train_config['dataset']['name'].lower()}_eval_{config['dataset']['name'].lower()}.npy"
    filepath_eval= entropy_folder / f"EVAL_train_{train_config['dataset']['name'].lower()}_eval_{config['dataset']['name'].lower()}.npy"

    np.save(filepath_ood, np.array(ood_entropy))
    np.save(filepath_eval, np.array(eval_entropy))


##############################################################################################################################################################

def acc_ensemble(experiments_models, train_config, config, folders, folder_ensemble, which_split, which_dataset=None):

    print("Evaluation using ensemble ...")

    if which_dataset is None: 
        which_dataset = which_split
    else:
        which_dataset = which_dataset + "_" + which_split

    folder_ensemble = folder_ensemble.parent / "all-proba"
    folder_ensemble.mkdir(exist_ok=True, parents=True)

    # get the evaluation data
    eval_loader = eval_cae.get_setup(config, folders, which_split=which_split, nbr_of_samples_per_class=eval_cae.SAMPLES_PER_CLASS_FOR_OOD[config["dataset"]["name"]], load_model=False, print_dataset=False)

    # keep track
    accurate_certain = defaultdict(int)
    accurate_uncertain = defaultdict(int)
    inaccurate_certain = defaultdict(int)
    inaccurate_uncertain = defaultdict(int)

    # certainty thresholds to consider
    threshold_to_consider = np.linspace(start=0.0, stop=1.0, num=50, endpoint=False) 

    # make sure we are in eval mode
    for current_model in experiments_models.values():
        current_model.eval()

    # do not keep track of gradients
    with torch.no_grad():

        # for each batch of training images
        for batch in eval_loader:

            # push to gpu
            input_images = batch["image"].to(config["device"])
            gts = batch["gt"].numpy()

            # keep track of several iterations
            keep_track_predictions = []

            for current_model in experiments_models.values():

                # encode the images
                prediction = current_model(input_images)["prediction"].cpu()

                # make them  one hot encoded vectors
                softmax_prediction = torch.nn.functional.softmax(prediction, dim=1)

                # keep track
                keep_track_predictions.append(softmax_prediction.numpy())

            # make sure its a numpy array of the correct shape
            # (nbr_trials, batch_size, nbr_classes)
            keep_track_predictions = np.stack(keep_track_predictions)
            # (batch_size, nbr_trials, nbr_classes)
            keep_track_predictions = np.transpose(keep_track_predictions, [1,0,2])

            # mean over the number of trials to get for each batch element a list
            # mean probability vector of values between 0 and 1
            class_probability = keep_track_predictions.mean(axis=1)

            # get the class with most counts as prediction
            prediction = class_probability.argmax(axis=1)
            prediction = np.array([int(x) for x in prediction])

            # entropy, normalize by log(nbr_classes) to get values between 0 and 1
            # log is the natural logarithm as used by stats.entropy as well
            uncertainty = stats.entropy(class_probability, axis=1) / np.log(current_model.nbr_classes)

            # for each threshold
            for threshold in threshold_to_consider:

                # get only the predictions for which the probability is high enough
                certain_mask = uncertainty < threshold
                uncertain_mask = uncertainty >= threshold

                accurate_certain[threshold] += (prediction[certain_mask] == gts[certain_mask]).sum()
                accurate_uncertain[threshold] += (prediction[uncertain_mask] == gts[uncertain_mask]).sum()
                inaccurate_certain[threshold] += (prediction[certain_mask] != gts[certain_mask]).sum()
                inaccurate_uncertain[threshold] += (prediction[uncertain_mask] != gts[uncertain_mask]).sum()


    utils.compute_scores(
        accurate_certain, 
        accurate_uncertain, 
        inaccurate_certain, 
        inaccurate_uncertain, 
        threshold_to_consider, 
        which_dataset, 
        {"data-all-proba":folder_ensemble},
        idx_recursion=-1 ,
        which_classifier=None,
        fix_dropout_mask=None,
    )

##############################################################################################################################################################

def ood_ensemble_experiments(ensemble_experiments):

    assert len(ensemble_experiments) == 10

    # keep track of the models for each experiments
    experiments_models = dict()

    # for each experiment
    for experiment in ensemble_experiments:
            
        # get the path to all the folder and files
        folders = eval_cae.get_folder(experiment)

        # load the config file
        train_config = toml.load(folders["config"])
        config = toml.load(folders["config"])
        assert config["model"]["dropout"] == 0.0

        # define the device
        train_config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # get the training data and the model
        _, model = eval_cae.get_setup(config, folders, which_split=config["dataset"]["split"], load_model=True, nbr_of_samples_per_class=config["dataset"]["nbr_of_samples_per_class"])

        # keep track of them for later
        experiments_models[experiment] = model

    
    folder_ensemble = [x.split("_")[0] for x in ensemble_experiments] 
    folder_ensemble = "_".join(folder_ensemble)
    folder_ensemble = "ensemble_" + folder_ensemble
    folder_ensemble = Path("results") / folder_ensemble / "data" / "ood"
    folder_ensemble.mkdir(exist_ok=True, parents=True)
    (folder_ensemble.parent.parent / "logs").mkdir(exist_ok=True)
    utils.write_toml_to_file(config, folder_ensemble.parent.parent / "cfg.toml")

    if config["dataset"]["name"].lower()  == "mnist":

        acc_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test")

        config["dataset"]["name"] = "fashion"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-fashion")
        config["dataset"]["name"] = "omniglot"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-omniglot")
        config["dataset"]["name"] = "cifar10"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-cifar10")
        config["dataset"]["name"] = "svhn"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-svhn")


    elif train_config["dataset"]["name"].lower()  == "fashion":

        acc_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test")

        config["dataset"]["name"] = "mnist"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-mnist")
        config["dataset"]["name"] = "omniglot"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-omniglot")
        config["dataset"]["name"] = "cifar10"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-cifar10")
        config["dataset"]["name"] = "svhn"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-svhn")


    elif train_config["dataset"]["name"].lower()  == "cifar10":

        acc_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test")

        config["dataset"]["name"] = "places365"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-places365")
        config["dataset"]["name"] = "lsun"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="train", which_dataset="ood-lsun")
        config["dataset"]["name"] = "svhn"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-svhn")

    elif train_config["dataset"]["name"].lower()  == "svhn":

        acc_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test")

        config["dataset"]["name"] = "places365"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-places365")
        config["dataset"]["name"] = "lsun"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="train", which_dataset="ood-lsun")
        config["dataset"]["name"] = "cifar10"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-cifar10")
        config["dataset"]["name"] = "gtsrb"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-gtsrb")

    elif config["dataset"]["name"].lower() == "gtsrb":

        acc_ensemble(experiments_models, train_config=train_config,config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test")
        ood_ensemble(experiments_models, train_config=train_config,config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="ood")

        config["dataset"]["name"] = "svhn"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-svhn")
        config["dataset"]["name"] = "places365"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-places365")
        config["dataset"]["name"] = "cifar10"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-cifar10")
        config["dataset"]["name"] = "lsun"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="train", which_dataset="ood-lsun")


    elif config["dataset"]["name"].lower() == "sviro_uncertainty":

        acc_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test-adults")
        acc_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test-adults-and-objects")
        acc_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test-adults-and-seats")
        acc_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test-adults-and-seats-and-objects")
        acc_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test-objects")
        acc_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test-seats")

        config["dataset"]["name"] = "svhn"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-svhn")
        config["dataset"]["name"] = "places365"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-places365")
        config["dataset"]["name"] = "cifar10"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-cifar10")
        config["dataset"]["name"] = "lsun"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="train", which_dataset="ood-lsun")
        config["dataset"]["name"] = "gtsrb"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="test", which_dataset="ood-gtsrb")

        config["dataset"]["name"] = "sviro"
        config["dataset"]["factor"] = "tesla"
        ood_ensemble(experiments_models, train_config=train_config, config=config, folders=folders, folder_ensemble=folder_ensemble, which_split="train", which_dataset="ood-tesla")

    # reset the stdout with the original one
    # this is necessary when the train function is called several times
    # by another script
    sys.stdout = sys.stdout.end()

##############################################################################################################################################################

if __name__ == "__main__":

    # specify which gpu should be visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------

    ensemble_experiments = [



    ]

    ood_ensemble_experiments(ensemble_experiments)


