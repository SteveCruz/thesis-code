##############################################################################################################################################################
##############################################################################################################################################################
"""
Evaluate attractor autoencoder models after training.
This one is meant for the attractor autoencoders only.

Simply insert the experiments to be evaluated inside the following list.
The folders need to be located inside the results folder.

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
from pathlib import Path
from importlib import import_module
from collections import defaultdict
from scipy import stats

from openTSNE import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

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
import dataset

##############################################################################################################################################################
##############################################################################################################################################################

SAMPLES_PER_CLASS_FOR_OOD = {
    "fashion": 250,
    "mnist": 250,
    "gtsrb": 250,
    "cifar10": 250,
    "svhn": 250,
    "omniglot": 4,
    "places365": 7, 
    "lsun": 250,
    "sviro_uncertainty": 250,
    "sviro": 250,
    "orss": 250,
    "orss_and_synth_nocar": 250,
    "seats_and_people_nocar": 250,
}

##############################################################################################################################################################
##############################################################################################################################################################

def get_folder(experiment):

    folders = dict()
    folders["experiment"] = Path("results") / experiment
    folders["scripts"] = folders["experiment"] / "scripts"
    folders["logs"] = folders["experiment"] / "logs"
    folders["latent"] = folders["experiment"] / "data" / "latent"
    folders["checkpoint"] = folders["experiment"] / "checkpoints" 
    folders["images"] = folders["experiment"] / "images" / "eval"
    folders["images-latent"] = folders["experiment"] / "images" / "latent"
    folders["images-proba"] = folders["experiment"] / "images" / "proba"
    folders["images-ood"] = folders["experiment"] / "images" / "ood"
    folders["data-proba"] = folders["experiment"] / "data" / "proba"
    folders["data-all-proba"] = folders["experiment"] / "data" / "all-proba"
    folders["data-ood"] = folders["experiment"] / "data" / "ood"
    folders["data-entropy"] = folders["experiment"] / "data" / "entropy"
    folders["config"] = folders["experiment"] / "cfg.toml"

    # check if the folder exists
    if not folders["experiment"].is_dir():
        print("The specified experiment does not exist: Exit script.")
        sys.exit()

    folders["images"].mkdir(parents=True, exist_ok=True)
    folders["images-latent"].mkdir(parents=True, exist_ok=True)
    folders["images-proba"].mkdir(parents=True, exist_ok=True)
    folders["images-ood"].mkdir(parents=True, exist_ok=True)
    folders["data-proba"].mkdir(parents=True, exist_ok=True)
    folders["data-all-proba"].mkdir(parents=True, exist_ok=True)
    folders["data-ood"].mkdir(parents=True, exist_ok=True)
    folders["data-entropy"].mkdir(parents=True, exist_ok=True)
    folders["latent"].mkdir(parents=True, exist_ok=True)

    # save the console output to a file
    sys.stdout = utils.Tee(original_stdout=sys.stdout, file=folders["logs"] / "testing.log")

    return folders

##############################################################################################################################################################        

def get_setup(config, folders, which_split, load_model=False, nbr_of_samples_per_class=-1, print_dataset=False, augment_bc_train=False):

    # load data
    data_loader = dataset.create_dataset(
        which_dataset=config["dataset"]["name"], 
        which_factor=config["dataset"]["factor"], 
        img_size=config["dataset"]["img_size"], 
        which_split=which_split,
        make_scene_impossible=False,
        make_instance_impossible=False,
        augment=config["training"]["augment"] if augment_bc_train else False,
        batch_size=512, 
        shuffle=True, 
        nbr_of_samples_per_class=nbr_of_samples_per_class,
        print_dataset=print_dataset
    )

    if load_model:

        # get path to the loader script
        model_file = folders["scripts"] / "model"

        # replace the / by .
        string_model_file = str(model_file).replace("/",".")

        # load the model from the file
        model_file = import_module(string_model_file)

        # get the model definition
        model_def = getattr(model_file, "create_ae_model")

        # define model
        model = model_def(config).to(config["device"])

        # load the model weights
        checkpoint = torch.load(folders["checkpoint"] / "last_model.pth", map_location=config["device"])

        # apply the model weights
        model.load_state_dict(checkpoint) 

        return data_loader, model

    return data_loader

##############################################################################################################################################################

def get_data(model, config, data_loader, nbr_recursions, fix_dropout_mask):

    print("Creating latent space ...")

    # keep track of latent space
    mus = []
    labels = []

    # make sure we are in eval mode
    model.eval()

    # we do not need to keep track of gradients
    with torch.no_grad():

        # for each batch of images
        for batch in data_loader:

            # push to gpu
            input_images = batch["image"].to(config["device"])
            gt = batch["gt"].numpy()

            if fix_dropout_mask:
                model._define_and_fix_new_dropout_mask(input_images)
                output = model(input_images, random=False)
            else:
                output = model(input_images)

            for _ in range(nbr_recursions):
                
                if fix_dropout_mask:
                    output = model(output["xhat"], random=False) 
                else:
                    output = model(output["xhat"]) 
            
            latent = output["mu"].cpu().numpy()

            # keep track of latent space
            mus.extend(latent) 
            labels.extend(gt)

        # otherwise not useable
        mus = np.array(mus)
        labels = np.array(labels)

    return mus, labels

##############################################################################################################################################################

def dropout_reconstruction(model, config, folders, which_split, which_dataset, nbr_recursions, nbr_trials, fix_dropout_mask):

    # the same order of images for each run
    # also make sure to only get the samples used during training
    # i.e. the reduced version of the training dataset
    eval_loader = get_setup(config, folders, which_split=which_split, load_model=False, nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[config["dataset"]["name"]], print_dataset=False)

    # do not keep track of gradients
    with torch.no_grad():

        # for each batch of training images
        torch.manual_seed("0")
        for input_images in eval_loader:
            
            # push to gpu
            input_images = input_images["image"][0:16].to(config["device"])

            # get each image individually
            for idx_img, image in enumerate(input_images):

                # add the batch dimension
                image = image.unsqueeze(0)

                # keep track of the different noise reconstructions for this image
                keep_track_all = []

                for _ in range(nbr_trials):

                    # keep track of the recursions for this noisy image
                    keep_track_recons = []

                    # make a copy of the input images 
                    current_input_image = image.detach().clone()

                    # reconstruction noisy image
                    if fix_dropout_mask:
                        model._define_and_fix_new_dropout_mask(current_input_image)
                        model_output = model(current_input_image, random=False)
                    else:
                        model_output = model(current_input_image)

                    # keep track
                    keep_track_recons.append(current_input_image)
                    keep_track_recons.append(model_output["xhat"])

                    # for each recursion, we do the same
                    for _ in range(nbr_recursions):

                        if fix_dropout_mask:
                            model_output = model(model_output["xhat"], random=False)
                        else:
                            model_output = model(model_output["xhat"])

                        if nbr_recursions < 19:
                            keep_track_recons.append(model_output["xhat"])

                    # make sure we have a tensor and keep track of it
                    if nbr_recursions >=19:
                        keep_track_recons.append(model_output["xhat"])
                    keep_track_recons = torch.cat(keep_track_recons, dim=0)
                    keep_track_all.append(keep_track_recons)

                # plot all the reconstructions for all recursive steps
                if fix_dropout_mask:
                    text = f"dropout_{nbr_recursions}_recursions_fixedmask_{which_dataset}_{idx_img}"
                else:
                    text = f"dropout_{nbr_recursions}_recursions_{which_dataset}_{idx_img}"
                
                if nbr_recursions >=19:
                    utils.plot_progress(images=keep_track_all, save_folder=folders, nbr_epoch="", text=text, max_nbr_samples=nbr_trials, transpose=True)
                else:
                    utils.plot_progress(images=keep_track_all, save_folder=folders, nbr_epoch="", text=text, max_nbr_samples=nbr_recursions+2, transpose=False)

            return

##############################################################################################################################################################


def dropout_uncertainty_per_recursion(model, classifier_per_recursion, train_labels, train_config, config, folders, which_split, which_dataset, nbr_recursions, nbr_trials, fix_dropout_mask):

    # the same order of images for each run
    # also make sure to only get the samples used during training
    # i.e. the reduced version of the training dataset
    eval_loader = get_setup(config, folders, which_split=which_split, nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[config["dataset"]["name"]], load_model=False, print_dataset=False)

    # keep track
    accurate_certain = defaultdict(dict)
    accurate_uncertain = defaultdict(dict)
    inaccurate_certain = defaultdict(dict)
    inaccurate_uncertain = defaultdict(dict)

    for idx_recursion in range(0, nbr_recursions, 1):
        accurate_certain[idx_recursion] = defaultdict(int)
        accurate_uncertain[idx_recursion] = defaultdict(int)
        inaccurate_certain[idx_recursion] = defaultdict(int)
        inaccurate_uncertain[idx_recursion] = defaultdict(int)

    # get the unique training labels
    unique_training_labels = np.unique(train_labels)
    # sort them
    unique_training_labels = sorted(unique_training_labels)
    # map from 0 to nbr_classes-1 to the actual integer labels
    int_to_class_distribution = {idx:int(x) for idx, x in enumerate(unique_training_labels)}
    # number of classes
    nbr_classes = len(unique_training_labels)

    # certainty thresholds to consider
    threshold_to_consider = np.linspace(start=0.0, stop=1.0, num=50, endpoint=False)

    # make sure we are in eval mode
    model.eval()

    # do not keep track of gradients
    with torch.no_grad():

        # for each batch of training images
        for batch in eval_loader:
            
            # push to gpu
            input_images = batch["image"].to(config["device"])
            gts = batch["gt"].numpy()

            keep_track_nn_predictions = defaultdict(list)

            # for each trial
            for _ in range(nbr_trials):

                # for each recursion, we do the same
                for idx_recursion in range(0, nbr_recursions):

                    if fix_dropout_mask:
                        if idx_recursion == 0:
                            model._define_and_fix_new_dropout_mask(input_images)
                            model_output = model(input_images, random=False)
                        else:
                            model_output = model(model_output["xhat"], random=False)
                    else:
                        if idx_recursion == 0:
                            model_output = model(input_images)
                        else:
                            model_output = model(model_output["xhat"])

                    if idx_recursion in classifier_per_recursion:

                        # get the last latent space representation
                        latent_space_after_recursion = model_output["mu"].cpu().numpy()

                        # get the class distribution prediction based on the last latent space representation
                        # will be all 0 but at 1 place
                        prediction = classifier_per_recursion[idx_recursion].predict_proba(latent_space_after_recursion)

                        # keep track
                        keep_track_nn_predictions[idx_recursion].append(prediction)


            for idx_recursion in range(0, nbr_recursions, 1):

                # make sure its a numpy array of the correct shape
                # (nbr_trials, batch_size, nbr_classes)
                keep_track_nn_predictions[idx_recursion] = np.stack(keep_track_nn_predictions[idx_recursion])
                # (batch_size, nbr_trials, nbr_classes)
                keep_track_nn_predictions[idx_recursion] = np.transpose(keep_track_nn_predictions[idx_recursion], [1,0,2])

                # mean over the number of trials to get for each batch element a list
                # mean probability vector of values between 0 and 1
                class_probability = keep_track_nn_predictions[idx_recursion].mean(axis=1)

                # get the class with most counts as prediction
                prediction = class_probability.argmax(axis=1)
                prediction = np.array([int_to_class_distribution[x] for x in prediction])

                # entropy, normalize by log(nbr_classes) to get values between 0 and 1
                # log is the natural logarithm as used by stats.entropy as well
                uncertainty = stats.entropy(class_probability, axis=1) / np.log(nbr_classes)

                # for each threshold
                for threshold in threshold_to_consider:

                    # get only the predictions for which the probability is high enough
                    certain_mask = uncertainty < threshold
                    uncertain_mask = uncertainty >= threshold

                    accurate_certain[idx_recursion][threshold] += (prediction[certain_mask] == gts[certain_mask]).sum()
                    accurate_uncertain[idx_recursion][threshold] += (prediction[uncertain_mask] == gts[uncertain_mask]).sum()
                    inaccurate_certain[idx_recursion][threshold] += (prediction[certain_mask] != gts[certain_mask]).sum()
                    inaccurate_uncertain[idx_recursion][threshold] += (prediction[uncertain_mask] != gts[uncertain_mask]).sum()

    for idx_recursion in range(0, nbr_recursions, 1):
        utils.compute_scores(
            accurate_certain[idx_recursion], 
            accurate_uncertain[idx_recursion], 
            inaccurate_certain[idx_recursion], 
            inaccurate_uncertain[idx_recursion], 
            threshold_to_consider, 
            which_dataset, 
            folders,
            idx_recursion=idx_recursion,
            which_classifier=train_config["which_classifier"],
            fix_dropout_mask=fix_dropout_mask
        )

##############################################################################################################################################################


def dropout_ood_per_recursion(model, classifier_per_recursion, train_labels, train_config, config, folders, which_split, which_dataset, nbr_recursions, nbr_trials, fix_dropout_mask):

    # the same order of images for each run
    # also make sure to only get the samples used during training
    # i.e. the reduced version of the training dataset
    ood_loader = get_setup(config, folders, which_split=which_split, nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[config["dataset"]["name"]], load_model=False, print_dataset=False)

    if train_config["dataset"]["name"].lower() == "sviro_uncertainty":
        if "seats" in train_config["dataset"]["split"].lower():
            test_loader = get_setup(train_config, folders, which_split="test-adults-and-seats", nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[train_config["dataset"]["name"]], load_model=False, print_dataset=False)
        else:
            test_loader = get_setup(train_config, folders, which_split="test-adults", nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[train_config["dataset"]["name"]], load_model=False, print_dataset=False)
    else:
        test_loader = get_setup(train_config, folders, which_split="test", nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[train_config["dataset"]["name"]], load_model=False, print_dataset=False)

    print(f"Length dataloader  OOD: {len(ood_loader)}")
    print(f"Length dataloader Test: {len(test_loader)}")

    # keep track
    true_positive = defaultdict(dict)
    false_positive = defaultdict(dict)
    true_negative = defaultdict(dict)
    false_negative = defaultdict(dict)
    ood_entropy = defaultdict(list)
    eval_entropy = defaultdict(list)

    for idx_recursion in range(0, nbr_recursions, 1):
        true_positive[idx_recursion] = defaultdict(int)
        false_positive[idx_recursion] = defaultdict(int)
        true_negative[idx_recursion] = defaultdict(int)
        false_negative[idx_recursion] = defaultdict(int)

    # get the unique training labels
    unique_training_labels = np.unique(train_labels)
    # sort them
    unique_training_labels = sorted(unique_training_labels)
    # map from 0 to nbr_classes-1 to the actual integer labels
    int_to_class_distribution = {idx:int(x) for idx, x in enumerate(unique_training_labels)}
    # number of classes
    nbr_classes = len(unique_training_labels)

    # certainty thresholds to consider
    threshold_to_consider = np.linspace(start=0.0, stop=1.0, num=50, endpoint=False)

    # make sure we are in eval mode
    model.eval()

    # do not keep track of gradients
    with torch.no_grad():

        for idx_data_loader, data_loader in enumerate([test_loader, ood_loader]):

            # for each batch of training images
            for batch in data_loader:
                
                # push to gpu
                input_images = batch["image"].to(config["device"])

                keep_track_nn_predictions = defaultdict(list)

                # for each trial
                for _ in range(nbr_trials):

                    # for each recursion, we do the same
                    for idx_recursion in range(0, nbr_recursions):

                        if fix_dropout_mask:
                            if idx_recursion == 0:
                                model._define_and_fix_new_dropout_mask(input_images)
                                model_output = model(input_images, random=False)
                            else:
                                model_output = model(model_output["xhat"], random=False)
                        else:
                            if idx_recursion == 0:
                                model_output = model(input_images)
                            else:
                                model_output = model(model_output["xhat"])

                        if idx_recursion in classifier_per_recursion:

                            # get the last latent space representation
                            latent_space_after_recursion = model_output["mu"].cpu().numpy()

                            # get the class distribution prediction based on the last latent space representation
                            # will be a softmax probability distribution
                            prediction = classifier_per_recursion[idx_recursion].predict_proba(latent_space_after_recursion)

                            # keep track
                            keep_track_nn_predictions[idx_recursion].append(prediction)


                for idx_recursion in range(0, nbr_recursions, 1):

                    # make sure its a numpy array of the correct shape
                    # (nbr_trials, batch_size, nbr_classes)
                    keep_track_nn_predictions[idx_recursion] = np.stack(keep_track_nn_predictions[idx_recursion])
                    # (batch_size, nbr_trials, nbr_classes)
                    keep_track_nn_predictions[idx_recursion] = np.transpose(keep_track_nn_predictions[idx_recursion], [1,0,2])

                    # mean over the number of trials to get for each batch element a list
                    # mean probability vector of values between 0 and 1
                    class_probability = keep_track_nn_predictions[idx_recursion].mean(axis=1)

                    # get the class with most counts as prediction
                    prediction = class_probability.argmax(axis=1)
                    prediction = np.array([int_to_class_distribution[x] for x in prediction])

                    # entropy, normalize by log(nbr_classes) to get values between 0 and 1
                    # log is the natural logarithm as used by stats.entropy as well
                    uncertainty = stats.entropy(class_probability, axis=1) / np.log(nbr_classes)

                    if idx_data_loader == 0:
                        eval_entropy[idx_recursion].extend(uncertainty)
                    else:
                        ood_entropy[idx_recursion].extend(uncertainty)

                    # for each threshold
                    for threshold in threshold_to_consider:

                        # get only the predictions for which the probability is high enough
                        certain_mask = uncertainty < threshold
                        uncertain_mask = uncertainty >= threshold

                        if idx_data_loader == 0:
                            true_positive[idx_recursion][threshold] += certain_mask.sum()
                            false_negative[idx_recursion][threshold] += uncertain_mask.sum()
                        else:
                            true_negative[idx_recursion][threshold] += uncertain_mask.sum()
                            false_positive[idx_recursion][threshold] += certain_mask.sum()

    for idx_recursion in range(0, nbr_recursions, 1):
        utils.compute_ood_scores(
            true_positive[idx_recursion], 
            true_negative[idx_recursion], 
            false_positive[idx_recursion], 
            false_negative[idx_recursion], 
            threshold_to_consider, 
            which_dataset, 
            folders,
            idx_recursion=idx_recursion,
            which_classifier=train_config["which_classifier"],
            fix_dropout_mask=fix_dropout_mask
        )

        filepath_ood = folders["data-entropy"] / f"OOD_train_{train_config['dataset']['name'].lower()}_eval_{config['dataset']['name'].lower()}_recursion_{idx_recursion}_{train_config['which_classifier']}.npy"
        filepath_eval= folders["data-entropy"] / f"EVAL_train_{train_config['dataset']['name'].lower()}_eval_{config['dataset']['name'].lower()}_recursion_{idx_recursion}_{train_config['which_classifier']}.npy"

        np.save(filepath_ood, np.array(ood_entropy[idx_recursion]))
        np.save(filepath_eval, np.array(eval_entropy[idx_recursion]))

##############################################################################################################################################################

def analyse_model(which_split, model, classifier, train_loader, classifier_per_recursion, train_mu, train_labels, train_config, config, folders, nbr_recursions, nbr_trials, which_dataset=None, ood=False):

    if which_dataset is None: 
        which_dataset = which_split
    else:
        which_dataset = which_dataset + "_" + which_split

    print(f"Evluating against: {which_dataset}")

    result_dropout_uncertainty = None
        
    # dropout_reconstruction(model, config, folders, which_split, which_dataset, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, fix_dropout_mask=False)
    dropout_reconstruction(model, config, folders, which_split, which_dataset, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, fix_dropout_mask=True)

    if ood:
        # dropout_ood_per_recursion(model, classifier_per_recursion, train_labels, train_config, config, folders, which_split, which_dataset, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, fix_dropout_mask=False)
        dropout_ood_per_recursion(model, classifier_per_recursion, train_labels, train_config, config, folders, which_split, which_dataset, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, fix_dropout_mask=True)
    else:
        # dropout_uncertainty_per_recursion(model, classifier_per_recursion, train_labels, train_config, config, folders, which_split, which_dataset, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, fix_dropout_mask=False)
        dropout_uncertainty_per_recursion(model, classifier_per_recursion, train_labels, train_config, config, folders, which_split, which_dataset, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, fix_dropout_mask=True)

    return result_dropout_uncertainty

##############################################################################################################################################################

def mlp_accuracy(model, which_split, train_config, config, folders, nbr_trials, which_dataset=None):

    if which_dataset is None: 
        which_dataset = which_split
    else:
        which_dataset = which_dataset + "_" + which_split

    # get the evaluation data
    eval_loader = get_setup(config, folders, which_split=which_split, load_model=False)

     # keep track
    accurate_certain = defaultdict(int)
    accurate_uncertain = defaultdict(int)
    inaccurate_certain = defaultdict(int)
    inaccurate_uncertain = defaultdict(int)

    # certainty thresholds to consider
    threshold_to_consider = np.linspace(start=0.0, stop=1.0, num=50, endpoint=False)

    # make sure we are in eval mode
    model.eval()

    # make sure dropout is enabled
    model._enable_dropout()

    # do not keep track of gradients
    with torch.no_grad():

        # for each batch of training images
        for batch in eval_loader:
            
            # push to gpu
            input_images = batch["image"].to(config["device"])
            gts = batch["gt"].numpy()

            keep_track_nn_predictions = []

            # for each recursion, we do the same
            for _ in range(nbr_trials):

                # get the logits, i.e. model predictions
                prediction = model(input_images)["prediction"].cpu()

                # make them  one hot encodeed vectors
                softmax_prediction = torch.nn.functional.softmax(prediction, dim=1)

                # keep track
                keep_track_nn_predictions.append(softmax_prediction.numpy())

            # make sure its a numpy array of the correct shape
            # (nbr_trials, batch_size, nbr_classes)
            keep_track_nn_predictions = np.stack(keep_track_nn_predictions)
            # (batch_size, nbr_trials, nbr_classes)
            keep_track_nn_predictions = np.transpose(keep_track_nn_predictions, [1,0,2])

            # mean over the number of trials to get for each batch element a list
            # mean probability vector of values between 0 and 1
            class_probability = keep_track_nn_predictions.mean(axis=1)

            # get the class with most counts as prediction
            prediction = class_probability.argmax(axis=1)
            prediction = np.array([int(x) for x in prediction])

            # entropy, normalize by log(nbr_classes) to get values between 0 and 1
            # log is the natural logarithm as used by stats.entropy as well
            uncertainty = stats.entropy(class_probability, axis=1) / np.log(model.nbr_classes)

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
        folders,
        idx_recursion=-1 ,
        which_classifier=None,
        fix_dropout_mask=None,
    )

##############################################################################################################################################################

def mlp_ood(model, which_split, train_config, config, folders, nbr_trials, which_dataset=None):

    if which_dataset is None: 
        which_dataset = which_split
    else:
        which_dataset = which_dataset + "_" + which_split

    # get the evaluation data
    ood_loader = get_setup(config, folders, which_split=which_split, nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[config["dataset"]["name"]], load_model=False, print_dataset=False)

    if train_config["dataset"]["name"].lower() == "sviro_uncertainty":
        if "seats" in train_config["dataset"]["split"].lower():
            test_loader = get_setup(train_config, folders, which_split="test-adults-and-seats", nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[train_config["dataset"]["name"]], load_model=False, print_dataset=False)
        else:
            test_loader = get_setup(train_config, folders, which_split="test-adults", nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[train_config["dataset"]["name"]], load_model=False, print_dataset=False)
    else:
        test_loader = get_setup(train_config, folders, which_split="test", nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[train_config["dataset"]["name"]], load_model=False, print_dataset=False)


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
    model.eval()

    # make sure dropout is enabled
    model._enable_dropout()

    # do not keep track of gradients
    with torch.no_grad():

        for idx_data_loader, data_loader in enumerate([test_loader, ood_loader]):

            # for each batch of training images
            for batch in data_loader:
            
                # push to gpu
                input_images = batch["image"].to(config["device"])

                keep_track_nn_predictions = []

                # for each recursion, we do the same
                for _ in range(nbr_trials):

                    # get the logits, i.e. model predictions
                    prediction = model(input_images)["prediction"].cpu()

                    # make them  one hot encodeed vectors
                    softmax_prediction = torch.nn.functional.softmax(prediction, dim=1)

                    # keep track
                    keep_track_nn_predictions.append(softmax_prediction.numpy())

                # make sure its a numpy array of the correct shape
                # (nbr_trials, batch_size, nbr_classes)
                keep_track_nn_predictions = np.stack(keep_track_nn_predictions)
                # (batch_size, nbr_trials, nbr_classes)
                keep_track_nn_predictions = np.transpose(keep_track_nn_predictions, [1,0,2])

                # mean over the number of trials to get for each batch element a list
                # mean probability vector of values between 0 and 1
                class_probability = keep_track_nn_predictions.mean(axis=1)

                # get the class with most counts as prediction
                prediction = class_probability.argmax(axis=1)
                prediction = np.array([int(x) for x in prediction])

                # entropy, normalize by log(nbr_classes) to get values between 0 and 1
                # log is the natural logarithm as used by stats.entropy as well
                uncertainty = stats.entropy(class_probability, axis=1) / np.log(model.nbr_classes)

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
        folders,
        idx_recursion=-1,
        which_classifier=None,
        fix_dropout_mask=None,
    )

    filepath_ood = folders["data-entropy"] / f"OOD_train_{train_config['dataset']['name'].lower()}_eval_{config['dataset']['name'].lower()}.npy"
    filepath_eval= folders["data-entropy"] / f"EVAL_train_{train_config['dataset']['name'].lower()}_eval_{config['dataset']['name'].lower()}.npy"

    np.save(filepath_ood, np.array(ood_entropy))
    np.save(filepath_eval, np.array(eval_entropy))

##############################################################################################################################################################

def evaluate_experiments(experiments, nbr_recursions, nbr_trials, which_classifier):

    # for each experiment
    for experiment in experiments:
        
        # in case we do something overnight and an error occurs somewhere for some reason
        try:

            print("/"*77)
            print(experiment)
            print("/"*77)
                
            # get the path to all the folder and files
            folders = get_folder(experiment)

            # load the config file
            train_config = toml.load(folders["config"])
            train_config["which_classifier"] = which_classifier
            config = toml.load(folders["config"])

            # define the device
            train_config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            # get the training data and the model
            train_loader, model = get_setup(train_config, folders, which_split=train_config["dataset"]["split"], load_model=True, nbr_of_samples_per_class=train_config["dataset"]["nbr_of_samples_per_class"], augment_bc_train=True)

            if model.type in ["mlp", "conv"]:

                if train_config["dataset"]["name"].lower()  == "mnist":

                    mlp_accuracy(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

                    config["dataset"]["name"] = "fashion"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-fashion")
                    config["dataset"]["name"] = "omniglot"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-omniglot")
                    config["dataset"]["name"] = "cifar10"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-cifar10")
                    config["dataset"]["name"] = "svhn"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-svhn")

                elif train_config["dataset"]["name"].lower()  == "fashion":

                    mlp_accuracy(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

                    config["dataset"]["name"] = "mnist"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-mnist")
                    config["dataset"]["name"] = "omniglot"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-omniglot")
                    config["dataset"]["name"] = "cifar10"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-cifar10")
                    config["dataset"]["name"] = "svhn"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-svhn")

                elif train_config["dataset"]["name"].lower() == "cifar10":

                    mlp_accuracy(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

                    config["dataset"]["name"] = "places365"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-places365")
                    config["dataset"]["name"] = "lsun"
                    mlp_ood(model=model, which_split="train", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-lsun")
                    config["dataset"]["name"] = "svhn"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-svhn")

                elif train_config["dataset"]["name"].lower() == "svhn":

                    mlp_accuracy(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

                    config["dataset"]["name"] = "places365"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-places365")
                    config["dataset"]["name"] = "lsun"
                    mlp_ood(model=model, which_split="train", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-lsun")
                    config["dataset"]["name"] = "cifar10"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-cifar10")
                    config["dataset"]["name"] = "gtsrb"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-gtsrb")


                elif train_config["dataset"]["name"].lower() == "gtsrb":

                    mlp_accuracy(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

                    config["dataset"]["name"] = "svhn"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-svhn")
                    config["dataset"]["name"] = "places365"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-places365")
                    config["dataset"]["name"] = "cifar10"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-cifar10")
                    config["dataset"]["name"] = "lsun"
                    mlp_ood(model=model, which_split="train", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-lsun")


                elif train_config["dataset"]["name"].lower() == "sviro_uncertainty":

                    mlp_accuracy(model=model, which_split="test-adults", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
                    mlp_accuracy(model=model, which_split="test-adults-and-objects", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
                    mlp_accuracy(model=model, which_split="test-adults-and-seats", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
                    mlp_accuracy(model=model, which_split="test-adults-and-seats-and-objects", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
                    mlp_accuracy(model=model, which_split="test-objects", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
                    mlp_accuracy(model=model, which_split="test-seats", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
                    
                    config["dataset"]["name"] = "svhn"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-svhn")
                    config["dataset"]["name"] = "places365"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-places365")
                    config["dataset"]["name"] = "cifar10"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-cifar10")
                    config["dataset"]["name"] = "lsun"
                    mlp_ood(model=model, which_split="train", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-lsun")
                    config["dataset"]["name"] = "gtsrb"
                    mlp_ood(model=model, which_split="test", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-gtsrb")

                    config["dataset"]["name"] = "sviro"
                    config["dataset"]["factor"] = "tesla"
                    mlp_ood(model=model, which_split="train", train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-tesla")

            else: 

                # define the classifier
                classifier_per_recursion = dict()
                for idx_recursion in range(0, nbr_recursions, 1):

                    train_mu, train_labels = get_data(model, train_config, train_loader, nbr_recursions=idx_recursion, fix_dropout_mask=True)
                    if train_config["which_classifier"] == "knn":
                        classifier = KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1)
                    elif train_config["which_classifier"] == "mlp":
                        classifier = MLPClassifier(hidden_layer_sizes=(model.latent_dim), max_iter=2000)
                    elif train_config["which_classifier"] == "linear":
                        classifier = SGDClassifier(loss="modified_huber", penalty="l2", n_jobs=-1)
                    classifier.fit(train_mu, train_labels)
                    classifier_per_recursion[idx_recursion] = classifier

                classifier = classifier_per_recursion[0]

                # train the classifier
                classifier.fit(train_mu, train_labels)

                if train_config["dataset"]["name"].lower()  == "mnist":

                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)

                    config["dataset"]["name"] = "fashion"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-fashion", ood=True)
                    config["dataset"]["name"] = "omniglot"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-omniglot", ood=True)
                    config["dataset"]["name"] = "cifar10"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-cifar10", ood=True)
                    config["dataset"]["name"] = "svhn"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-svhn", ood=True)

                elif train_config["dataset"]["name"].lower()  == "fashion":

                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)

                    config["dataset"]["name"] = "mnist"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-mnist", ood=True)
                    config["dataset"]["name"] = "omniglot"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-omniglot", ood=True)
                    config["dataset"]["name"] = "cifar10"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-cifar10", ood=True)
                    config["dataset"]["name"] = "svhn"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-svhn", ood=True)

                elif train_config["dataset"]["name"].lower()  == "cifar10":

                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)

                    config["dataset"]["name"] = "places365"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-places365", ood=True)
                    config["dataset"]["name"] = "lsun"
                    analyse_model(which_split="train", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-lsun", ood=True)
                    config["dataset"]["name"] = "svhn"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-svhn", ood=True)

                elif train_config["dataset"]["name"].lower()  == "svhn":

                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)

                    config["dataset"]["name"] = "places365"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-places365", ood=True)
                    config["dataset"]["name"] = "lsun"
                    analyse_model(which_split="train", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-lsun", ood=True)
                    config["dataset"]["name"] = "cifar10"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-cifar10", ood=True)
                    config["dataset"]["name"] = "gtsrb"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-gtsrb", ood=True)

                elif train_config["dataset"]["name"].lower()  == "gtsrb":

                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)

                    # analyse_model(which_split="ood", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=True)

                    config["dataset"]["name"] = "svhn"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-svhn", ood=True)
                    config["dataset"]["name"] = "places365"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-places365", ood=True)
                    config["dataset"]["name"] = "cifar10"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-cifar10", ood=True)
                    config["dataset"]["name"] = "lsun"
                    analyse_model(which_split="train", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-lsun", ood=True)
                    
                
                elif train_config["dataset"]["name"].lower() == "sviro_uncertainty":

                    analyse_model(which_split="test-adults", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)
                    analyse_model(which_split="test-adults-and-objects", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)
                    analyse_model(which_split="test-adults-and-seats", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)
                    analyse_model(which_split="test-adults-and-seats-and-objects", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)
                    analyse_model(which_split="test-objects", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)
                    analyse_model(which_split="test-seats", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, ood=False)

                    config["dataset"]["name"] = "svhn"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-svhn", ood=True)
                    config["dataset"]["name"] = "places365"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-places365", ood=True)
                    config["dataset"]["name"] = "cifar10"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-cifar10", ood=True)
                    config["dataset"]["name"] = "lsun"
                    analyse_model(which_split="train", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-lsun", ood=True)
                    config["dataset"]["name"] = "gtsrb"
                    analyse_model(which_split="test", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-gtsrb", ood=True)
                
                    config["dataset"]["name"] = "sviro"
                    config["dataset"]["factor"] = "tesla"
                    analyse_model(which_split="train", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-tesla", ood=True)

                elif train_config["dataset"]["name"].lower() == "orss":

                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="test-Sharan", ood=False)

                    config["dataset"]["name"] = "orss"
                    config["dataset"]["factor"] = "X5" if config["dataset"]["factor"] == "Sharan" else "Sharan"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-X5", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-X5", ood=False)

                    config["dataset"]["name"] = "sviro"
                    config["dataset"]["factor"] = "tesla"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-tesla", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-tesla", ood=False)
                    
                    config["dataset"]["name"] = "sviro"
                    config["dataset"]["factor"] = "hilux"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-hilux", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-hilux", ood=False)
                    
                    config["dataset"]["name"] = "sviro"
                    config["dataset"]["factor"] = "tucson"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-tucson", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-tucson", ood=False)
                    
                    config["dataset"]["name"] = "sviro"
                    config["dataset"]["factor"] = "aclass"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-aclass", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-aclass", ood=False)

                elif train_config["dataset"]["name"].lower() in ["orss_and_synth_nocar", "seats_and_people_nocar", "sviro"]:

                    config["dataset"]["name"] = "orss"
                    config["dataset"]["factor"] = "X5"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-X5", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-X5", ood=False)

                    config["dataset"]["name"] = "orss"
                    config["dataset"]["factor"] = "Sharan"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-Sharan", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-Sharan", ood=False)

                    config["dataset"]["name"] = "sviro"
                    config["dataset"]["factor"] = "tesla"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-tesla", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-tesla", ood=False)
                    
                    config["dataset"]["name"] = "sviro"
                    config["dataset"]["factor"] = "hilux"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-hilux", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-hilux", ood=False)
                    
                    config["dataset"]["name"] = "sviro"
                    config["dataset"]["factor"] = "tucson"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-tucson", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-tucson", ood=False)
                    
                    config["dataset"]["name"] = "sviro"
                    config["dataset"]["factor"] = "aclass"
                    config["dataset"]["split"] = "all"
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="ood-aclass", ood=True)
                    analyse_model(which_split="all", model=model, classifier=classifier, train_loader=train_loader, classifier_per_recursion=classifier_per_recursion, train_mu=train_mu, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_recursions=nbr_recursions, nbr_trials=nbr_trials, which_dataset="eval-aclass", ood=False)


                # reset the stdout with the original one
                # this is necessary when the train function is called several times
                # by another script
                sys.stdout = sys.stdout.end()

        except KeyboardInterrupt as e:
            print("Interupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"An error occured while processing experiment {experiment}: {e} ")

##############################################################################################################################################################

if __name__ == "__main__":

    # specify which gpu should be visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # number of recursions to do
    # nbr_recursions = 200
    # nbr_recursions = 50
    # nbr_recursions = 29
    # nbr_recursions = 22
    # nbr_recursions = 19
    # nbr_recursions = 17
    # nbr_recursions = 12
    # nbr_recursions = 9
    # nbr_recursions = 6

    # number of trials per image to do
    # nbr_trials = 50
    # nbr_trials = 20
    # nbr_trials = 10

    # which classifier to use
    # which_classifier = "knn"
    # which_classifier = "mlp"
    # which_classifier = "linear"

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------

    # list of all experiments to check
    experiments = [

        
    ]

    # evaluate all the experiments from the list
    evaluate_experiments(experiments, nbr_recursions, nbr_trials, which_classifier)

