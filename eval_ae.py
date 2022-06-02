##############################################################################################################################################################
##############################################################################################################################################################
"""
Evaluate the models.
This is primarily meant to evaluate uncertainty estimation.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import os
import sys
import toml
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from importlib import import_module
from scipy import stats

from openTSNE import TSNE
from umap import UMAP
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

import torchvision.transforms.functional as TF

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

# width as measured in inkscape
PLT_WIDTH = 6.053 # full width image for thesis
PLT_HEIGHT = PLT_WIDTH / 1.618

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

def get_configs(folders):

    # load the config file
    train_config = toml.load(folders["config"])
    test_config = toml.load(folders["config"])

    # define the device
    train_config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return train_config, test_config

##############################################################################################################################################################        

def get_setup(config, folders, which_split, load_model=False, nbr_of_samples_per_class=-1, print_dataset=False):

    # load data
    data_loader = dataset.create_dataset(
        which_dataset=config["dataset"]["name"], 
        which_factor=config["dataset"]["factor"], 
        img_size=config["dataset"]["img_size"], 
        which_split=which_split,
        make_scene_impossible=False,
        make_instance_impossible=False,
        augment=False,
        batch_size=128, 
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

def get_data(model, config, data_loader, flipped=False):

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

            if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                input_type = "real" if data_loader.dataset.__class__.__name__.lower() in ["orss", "ticam"] else "synth"
                output = model(input_images, input_type=input_type)["mu"]
            else:
                output = model(input_images)["mu"]

            
            latent = output.cpu().numpy()

            # keep track of latent space
            mus.extend(latent) 
            labels.extend(gt)

            if flipped:

                flipped_input_images = torch.stack([TF.hflip(x) for x in input_images])
                curr_flipped_labels = [data_loader.dataset.label_str_to_int[data_loader.dataset.int_to_label_str[x][::-1]] for x in gt]

                if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                    flipped_output = model(flipped_input_images, input_type=input_type)["mu"]
                else:
                    flipped_output = model(flipped_input_images)["mu"]

                flipped_latent = flipped_output.cpu().numpy()

                # keep track of latent space
                mus.extend(flipped_latent) 
                labels.extend(curr_flipped_labels)

            if "real_image" in batch:
                input_images = batch["real_image"].to(config["device"])
                gt = batch["gt"].numpy()

                if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                    input_type = "real" if data_loader.dataset.__class__.__name__.lower() in ["orss", "ticam"] else "synth"
                    output = model(input_images, input_type=input_type)["mu"]
                else:
                    output = model(input_images)["mu"]

                
                latent = output.cpu().numpy()

                # keep track of latent space
                mus.extend(latent) 
                labels.extend(gt)

                if flipped:

                    flipped_input_images = torch.stack([TF.hflip(x) for x in input_images])
                    curr_flipped_labels = [data_loader.dataset.label_str_to_int[data_loader.dataset.int_to_label_str[x][::-1]] for x in gt]

                    if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                        flipped_output = model(flipped_input_images, input_type=input_type)["mu"]
                    else:
                        flipped_output = model(flipped_input_images)["mu"]

                    flipped_latent = flipped_output.cpu().numpy()

                    # keep track of latent space
                    mus.extend(flipped_latent) 
                    labels.extend(curr_flipped_labels)

        # otherwise not useable
        mus = np.array(mus)
        labels = np.array(labels)

    return mus, labels

##############################################################################################################################################################


def project_train_and_test(train_mu, train_labels, eval_mu, eval_labels, int_to_label_str, folders, which_split):

    # define folder to plot result
    save_folder = folders["experiment"] / "images" / "latent"
    save_folder.mkdir(exist_ok=True)

    all_projection_methods = {
        "tsne" : TSNE,
        "umap" : UMAP,
        "pca" : PCA,
    }

    for projection_name, projection_method in all_projection_methods.items():

        # compute the tsne of the data using all cores
        if projection_name == "umap":
            transformation = projection_method(n_jobs=-1).fit(train_mu)
            train_embedding = transformation.embedding_
            test_embedding = transformation.transform(eval_mu)
        elif projection_name == "tsne":
            train_embedding = projection_method(n_jobs=-1).fit(train_mu)
            test_embedding = train_embedding.transform(eval_mu)
        elif projection_name == "pca":
            transformation = projection_method()
            transformation.fit(train_mu)
            train_embedding = transformation.transform(train_mu)
            test_embedding = transformation.transform(eval_mu)

        # get the unique class labels available
        classes = np.unique(train_labels)

        # keep track of the scatter plots
        train_scatters = []
        test_scatters = []

        # create plot
        fig, ax = plt.subplots(1,1)

        # plot train samples first
        for current_class in classes:
            
            str_current_class = int_to_label_str[current_class]
            train_mask = train_labels == current_class
            current_train_embedding = train_embedding[train_mask]
            sc_train = ax.scatter(current_train_embedding[:, 0], current_train_embedding[:, 1], alpha=0.50, s=17, edgecolor='white', linewidth=0.33, rasterized=True, label=str_current_class, marker=".")
            train_scatters.append(sc_train)

        # remove the ticks from the plot as they are meaningless
        ax.axis('equal')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.box(False)
        plt.grid(True, color="0.9", linestyle='-', linewidth=1)

        # adjust the space between the marker and the text
        # and the vertical offset of the marker to center with text
        if len(classes) < 15:
            plt.legend(fontsize=5, scatteryoffsets=[0.48], handletextpad=0.1)

        # fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
        fig.savefig(save_folder / (f"train_vs_{which_split}_{projection_name}_notest.png") ,bbox_inches='tight')
        fig.savefig(save_folder / (f"train_vs_{which_split}_{projection_name}_notest.pdf") ,bbox_inches='tight')

        # we want to make sure to plot all test samples after the train samples are done
        # this way for sure all test samples will be on top of the train samples
        for idx, current_class in enumerate(classes):

            str_current_class = int_to_label_str[current_class]
            test_mask = eval_labels == current_class
            current_test_embedding = test_embedding[test_mask]
            sc_test = ax.scatter(current_test_embedding[:, 0], current_test_embedding[:, 1], alpha=0.75, s=25, edgecolor='white', linewidth=0.5, rasterized=True, label=str_current_class, color=train_scatters[idx].get_facecolors()[0].tolist(), marker="+")
            test_scatters.append(sc_test)

        # remove the ticks from the plot as they are meaningless
        ax.axis('equal')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.box(False)
        plt.grid(True, color="0.9", linestyle='-', linewidth=1)

        # save the same figure without the test samples
        fig.savefig(save_folder / (f"train_vs_{which_split}_{projection_name}.png") ,bbox_inches='tight')
        fig.savefig(save_folder / (f"train_vs_{which_split}_{projection_name}.pdf") ,bbox_inches='tight')

        plt.close(fig=fig)

    print("="*37)


##############################################################################################################################################################

def project_synth_and_real(model, train_config, folders):

    # define folder to plot result
    save_folder = folders["experiment"] / "images" / "latent"
    save_folder.mkdir(exist_ok=True)

    real_config, synth_config = get_configs(folders)

    real_config["dataset"]["name"] = "orss"
    real_config["dataset"]["factor"] = "X5"
    real_config["dataset"]["split"] = "all"
    real_loader = get_setup(real_config, folders, which_split="all", load_model=False)

    synth_config["dataset"]["name"] = "seats_and_people_nocar"
    synth_config["dataset"]["factor"] = ""
    synth_config["dataset"]["split"] = "all"
    synth_loader = get_setup(synth_config, folders, which_split="all", load_model=False)
    real_mu, _, synth_mu, synth_labels = latent_spaces(model, train_config, real_loader, synth_loader)

    all_projection_methods = {
        "tsne" : TSNE,
        "umap" : UMAP,
        "pca" : PCA,
    }

    # combine training and test mus
    together_mu = np.concatenate((real_mu, synth_mu))

    for projection_name, projection_method in all_projection_methods.items():

        # compute the tsne of the data using all cores
        if projection_name == "umap":
            transformation = projection_method(n_jobs=-1).fit(together_mu)
            together_embedding = transformation.embedding_
        elif projection_name == "tsne":
            together_embedding = projection_method(n_jobs=-1).fit(together_mu)
        elif projection_name == "pca":
            transformation = projection_method()
            transformation.fit(together_mu)
            together_embedding = transformation.transform(together_mu)

        # split the data again in test and train
        train_embedding = together_embedding[0:real_mu.shape[0]]
        test_embedding = together_embedding[real_mu.shape[0]:]

        # create plot
        fig, ax = plt.subplots(1,1)

            
        ax.scatter(train_embedding[:, 0], train_embedding[:, 1], alpha=0.50, s=17, edgecolor='white', linewidth=0.5, rasterized=True)
        ax.scatter(test_embedding[:, 0], test_embedding[:, 1], alpha=0.50, s=17, edgecolor='white', linewidth=0.5, rasterized=True, marker="X")

        # remove the ticks from the plot as they are meaningless
        ax.axis('equal')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.box(False)
        plt.grid(True, color="0.9", linestyle='-', linewidth=1)

        fig.savefig(save_folder / (f"real_vs_synth_{projection_name}.png") ,bbox_inches='tight')
        fig.savefig(save_folder / (f"real_vs_synth_{projection_name}.pdf") ,bbox_inches='tight')

        plt.close(fig=fig)

    print("="*37)

##############################################################################################################################################################

def reconstruction(model, data_loader, config, folders, split, train_mu=None, nbr_nearest_neighbours=10):

    if train_mu is not None:
        neighborhood = NearestNeighbors(n_neighbors=nbr_nearest_neighbours).fit(train_mu)

    # make sure we are in eval mode
    model.eval()

    # do not keep track of gradients
    with torch.no_grad():

        # for each batch of training images
        torch.manual_seed("0")
        for idx, input_images in enumerate(data_loader):
            
            # push to gpu
            input_images = input_images["image"].to(config["device"])

            if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                input_type = "real" if data_loader.dataset.__class__.__name__.lower() in ["orss", "ticam"] else "synth"
                model_output = model(input_images, input_type=input_type)
            else:
                model_output = model(input_images)

            # plot the samples
            utils.plot_progress(images=[input_images, model_output["xhat"]], save_folder=folders, nbr_epoch=idx, text=split+"_batch")

            # interpolation
            keep_track = [input_images]
            input_examples = model_output["mu"][0:16]
            output_examples = model_output["mu"][1:17]

            for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

                curr_example = (1-alpha)*input_examples + alpha*output_examples
                model_output = model.decode(curr_example)

                keep_track.append(model_output) 

            # plot the samples
            utils.plot_progress(images=keep_track, save_folder=folders, nbr_epoch=idx, text=f"{split}_interpolation_batch")

            
            if train_mu is not None:
                if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                    input_type = "real" if data_loader.dataset.__class__.__name__.lower() in ["orss", "ticam"] else "synth"
                    model_output = model(input_images, input_type=input_type)["mu"]
                else:
                    model_output = model(input_images)["mu"]

                _, indices = neighborhood.kneighbors(model_output.cpu().numpy())
                nn = train_mu[indices]

                if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                    input_type = "real" if data_loader.dataset.__class__.__name__.lower() in ["orss", "ticam"] else "synth"
                    model_output = model.decode(torch.from_numpy(nn).squeeze().to(input_images.device), input_type=input_type)
                else:
                    model_output = model.decode(torch.from_numpy(nn).squeeze().to(input_images.device))

                # plot the samples
                utils.plot_progress(images=[input_images, model_output], save_folder=folders, nbr_epoch=idx, text=split+"_nn_batch")
            
            # nn reconstructions
            if train_mu is not None:
                if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                    input_type = "real" if data_loader.dataset.__class__.__name__.lower() in ["orss", "ticam"] else "synth"
                    model_output = model(input_images, input_type=input_type)
                else:
                    model_output = model(input_images)

                _, indices = neighborhood.kneighbors(model_output["mu"].cpu().numpy())
                nn = train_mu[indices]
                
                keep_track = [input_images, model_output["xhat"]]

                for idx_nearest_neighbour in range(nbr_nearest_neighbours):

                    current_nearest_neighbour = nn[:,idx_nearest_neighbour,:]

                    if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                        input_type = "real" if data_loader.dataset.__class__.__name__.lower() in ["orss", "ticam"] else "synth"
                        model_output = model.decode(torch.from_numpy(current_nearest_neighbour).squeeze().to(input_images.device), input_type=input_type)
                    else:
                        model_output = model.decode(torch.from_numpy(current_nearest_neighbour).squeeze().to(input_images.device))

                    keep_track.append(model_output)

                # plot the samples
                utils.plot_progress(images=keep_track, save_folder=folders, nbr_epoch=idx, text=f"{split}_{nbr_nearest_neighbours}_nn_batch")

            if idx==9:
                return

##############################################################################################################################################################

def dropout_reconstruction(model, config, folders, which_split, which_dataset, nbr_trials):

    model.eval()
    model._enable_dropout()

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
            input_images = input_images["image"][0:6].to(config["device"])

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
                    model_output = model(current_input_image)

                    # keep track
                    keep_track_recons.append(current_input_image)
                    keep_track_recons.append(model_output["xhat"])

                    keep_track_recons = torch.cat(keep_track_recons, dim=0)
                    keep_track_all.append(keep_track_recons)

                # plot all the reconstructions for all recursive steps
                text = f"dropout_{which_dataset}_{idx_img}"
                
                utils.plot_progress(images=keep_track_all, save_folder=folders, nbr_epoch="", text=text, transpose=True)

            return

##############################################################################################################################################################

def dropout_uncertainty(model, classifier, train_labels, train_config, config, folders, which_split, nbr_trials, which_dataset=None):

    if which_dataset is None: 
        which_dataset = which_split
    else:
        which_dataset = which_dataset + "_" + which_split

    dropout_reconstruction(model, config, folders, which_split, which_dataset, nbr_trials)

    # the same order of images for each run
    # also make sure to only get the samples used during training
    # i.e. the reduced version of the training dataset
    eval_loader = get_setup(config, folders, which_split=which_split, nbr_of_samples_per_class=SAMPLES_PER_CLASS_FOR_OOD[config["dataset"]["name"]], load_model=False, print_dataset=False)

    # keep track
    accurate_certain = defaultdict(int)
    accurate_uncertain = defaultdict(int)
    inaccurate_certain = defaultdict(int)
    inaccurate_uncertain = defaultdict(int)

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
    model._enable_dropout()

    # do not keep track of gradients
    with torch.no_grad():

        # for each batch of training images
        for batch in eval_loader:
            
            # push to gpu
            input_images = batch["image"].to(config["device"])
            gts = batch["gt"].numpy()

            keep_track_nn_predictions = []

            # for each trial
            for _ in range(nbr_trials):

                model_output = model(input_images)

                # get the last latent space representation
                latent_space = model_output["mu"].cpu().numpy()

                # get the class distribution prediction based on the last latent space representation
                # will be all 0 but at 1 place
                prediction = classifier.predict_proba(latent_space)

                # keep track
                keep_track_nn_predictions.append(prediction)

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
            prediction = np.array([int_to_class_distribution[x] for x in prediction])

            # entropy, normalize by log(nbr_classes) to get values between 0 and 1
            # log is the natural logarithm as used by stats.entropy as well
            uncertainty = stats.entropy(class_probability, axis=1) / np.log(nbr_classes)

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
        idx_recursion=-1,
        which_classifier=train_config["which_classifier"],
        fix_dropout_mask=False
    )

##############################################################################################################################################################


def dropout_ood(model, classifier, train_labels, train_config, config, folders, which_split, nbr_trials, which_dataset=None):

    if which_dataset is None: 
        which_dataset = which_split
    else:
        which_dataset = which_dataset + "_" + which_split

    dropout_reconstruction(model, config, folders, which_split, which_dataset, nbr_trials)

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

    # keep track
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    true_negative = defaultdict(int)
    false_negative = defaultdict(int)
    ood_entropy = []
    eval_entropy = []

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
    model._enable_dropout()

    # do not keep track of gradients
    with torch.no_grad():

        for idx_data_loader, data_loader in enumerate([test_loader, ood_loader]):

            # for each batch of training images
            for batch in data_loader:
                
                # push to gpu
                input_images = batch["image"].to(config["device"])

                keep_track_nn_predictions = []

                # for each trial
                for _ in range(nbr_trials):

                    model_output = model(input_images)

                    # get the last latent space representation
                    latent_space = model_output["mu"].cpu().numpy()

                    # get the class distribution prediction based on the last latent space representation
                    # will be a softmax probability distribution
                    prediction = classifier.predict_proba(latent_space)

                    # keep track
                    keep_track_nn_predictions.append(prediction)

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
                prediction = np.array([int_to_class_distribution[x] for x in prediction])

                # entropy, normalize by log(nbr_classes) to get values between 0 and 1
                # log is the natural logarithm as used by stats.entropy as well
                uncertainty = stats.entropy(class_probability, axis=1) / np.log(nbr_classes)

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
        which_classifier=train_config["which_classifier"],
        fix_dropout_mask=False
    )

    filepath_ood = folders["data-entropy"] / f"OOD_train_{train_config['dataset']['name'].lower()}_eval_{config['dataset']['name'].lower()}_{train_config['which_classifier']}.npy"
    filepath_eval= folders["data-entropy"] / f"EVAL_train_{train_config['dataset']['name'].lower()}_eval_{config['dataset']['name'].lower()}_{train_config['which_classifier']}.npy"

    np.save(filepath_ood, np.array(ood_entropy))
    np.save(filepath_eval, np.array(eval_entropy))

##############################################################################################################################################################

def ood_inspection(model, classifier, train_labels, train_config, config, folders, nbr_trials):

    if train_config["dataset"]["name"].lower()  == "mnist":

        dropout_uncertainty(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

        config["dataset"]["name"] = "fashion"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-fashion")
        config["dataset"]["name"] = "omniglot"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-omniglot")
        config["dataset"]["name"] = "cifar10"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-cifar10")
        config["dataset"]["name"] = "svhn"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-svhn")

    elif train_config["dataset"]["name"].lower()  == "fashion":

        dropout_uncertainty(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

        config["dataset"]["name"] = "mnist"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-mnist")
        config["dataset"]["name"] = "omniglot"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-omniglot")
        config["dataset"]["name"] = "cifar10"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-cifar10")
        config["dataset"]["name"] = "svhn"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-svhn")

    elif train_config["dataset"]["name"].lower()  == "cifar10":

        dropout_uncertainty(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

        config["dataset"]["name"] = "places365"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-places365")
        config["dataset"]["name"] = "lsun"
        dropout_ood(which_split="train", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-lsun")
        config["dataset"]["name"] = "svhn"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-svhn")

    elif train_config["dataset"]["name"].lower()  == "svhn":

        dropout_uncertainty(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

        config["dataset"]["name"] = "places365"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-places365")
        config["dataset"]["name"] = "lsun"
        dropout_ood(which_split="train", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-lsun")
        config["dataset"]["name"] = "cifar10"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-cifar10")
        config["dataset"]["name"] = "gtsrb"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-gtsrb")

    elif train_config["dataset"]["name"].lower()  == "gtsrb":

        dropout_uncertainty(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

        config["dataset"]["name"] = "svhn"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-svhn")
        config["dataset"]["name"] = "places365"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-places365")
        config["dataset"]["name"] = "cifar10"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-cifar10")
        config["dataset"]["name"] = "lsun"
        dropout_ood(which_split="train", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-lsun")
        
    
    elif train_config["dataset"]["name"].lower() == "sviro_uncertainty":

        dropout_uncertainty(which_split="test-adults", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
        dropout_uncertainty(which_split="test-adults-and-objects", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
        dropout_uncertainty(which_split="test-adults-and-seats", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
        dropout_uncertainty(which_split="test-adults-and-seats-and-objects", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
        dropout_uncertainty(which_split="test-objects", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)
        dropout_uncertainty(which_split="test-seats", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials)

        config["dataset"]["name"] = "svhn"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-svhn")
        config["dataset"]["name"] = "places365"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-places365")
        config["dataset"]["name"] = "cifar10"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-cifar10")
        config["dataset"]["name"] = "lsun"
        dropout_ood(which_split="train", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-lsun")
        config["dataset"]["name"] = "gtsrb"
        dropout_ood(which_split="test", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-gtsrb")
    
        config["dataset"]["name"] = "sviro"
        config["dataset"]["factor"] = "tesla"
        dropout_ood(which_split="train", model=model, classifier=classifier, train_labels=train_labels, train_config=train_config, config=config, folders=folders, nbr_trials=nbr_trials, which_dataset="ood-tesla")


##############################################################################################################################################################

def reconstructions(model, train_loader, eval_loader, train_config, folders, train_mu=None):

    # reconstruct training and test samples
    reconstruction(model, train_loader, train_config, folders, "train", train_mu)
    reconstruction(model, eval_loader, train_config, folders, "test", train_mu)

def latent_spaces(model, train_config, train_loader, eval_loader):

    train_mu, train_labels = get_data(model, train_config, train_loader, flipped=True)
    eval_mu, eval_labels = get_data(model, train_config, eval_loader, flipped=False)

    tmp_eval_mu = []
    tmp_eval_labels = []

    for x,y in zip(eval_mu, eval_labels):
        # if "2" in train_loader.dataset.int_to_label_str[y]:
        #     continue
        if y in train_labels:
            tmp_eval_mu.append(x)
            tmp_eval_labels.append(y)

    eval_mu = np.array(tmp_eval_mu)
    eval_labels = np.array(tmp_eval_labels)

    return train_mu, train_labels, eval_mu, eval_labels

def classifications(train_mu, train_labels, eval_mu, eval_labels, train_config, folders, which_split):

    classifiers = {
        "knn-1":KNeighborsClassifier(n_neighbors=1, n_jobs=-1),
        "mlp":MLPClassifier(hidden_layer_sizes=(train_config["model"]["dimension"]), max_iter=10000, alpha=0.001),
        "linear":SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001, n_jobs=-1),
    }

    # for each classifier
    for name, classifier in classifiers.items():

        # train the classifier
        classifier.fit(train_mu, train_labels)

        # evaluate the classifier on this data
        score = classifier.score(eval_mu, eval_labels)

        # save accuracy
        utils.save_accuracy(Path(folders["logs"]) / f'test_accuracy_{which_split}_{name}.performance', score)
        
        print(f"[Testing] {name} - \tAccuracy {which_split}: {100*score:.1f}% (Nbr-Train: {train_labels.shape[0]}, Nbr-Eval: {eval_labels.shape[0]})")

    return classifiers

def uncertainties(model, train_config, folders, train_labels, nbr_trials, which_classifier, classifiers):

    if nbr_trials is not None and which_classifier is not None and which_classifier in classifiers:
                
        train_config["which_classifier"] = which_classifier
        config = toml.load(folders["config"])
        config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        ood_inspection(model, classifiers[which_classifier], train_labels, train_config, config, folders, nbr_trials)

    else:
        print("[WARNING] Skipping uncertainty and OOD evaluation.")

def analyse_model(model, train_config, train_loader, test_config, eval_loader, folders):

    if test_config["dataset"]["factor"]:
        which_split = test_config["dataset"]["name"] + "-" + test_config["dataset"]["factor"] + "-" + test_config["dataset"]["split"]
    else:
        which_split = test_config["dataset"]["name"] + "-" + test_config["dataset"]["split"]


    train_mu, train_labels, eval_mu, eval_labels = latent_spaces(model, train_config, train_loader, eval_loader)
    classifiers = classifications(train_mu, train_labels, eval_mu, eval_labels, train_config, folders, which_split)

    reconstructions(model, train_loader, eval_loader, train_config, folders)
    reconstructions(model, train_loader, eval_loader, train_config, folders, train_mu=train_mu)

    uncertainties(model, train_config, folders, train_labels, nbr_trials, which_classifier, classifiers)

    # plot the latent space
    project_train_and_test(train_mu, train_labels, eval_mu, eval_labels, train_loader.dataset.int_to_label_str, folders, "test")
    project_synth_and_real(model, train_config, folders)


##############################################################################################################################################################

def evaluate_experiments(experiments, nbr_trials, which_classifier):
    
    # for each experiment
    for experiment in experiments:
        
        # in case we do something overnight and an error occurs somewhere for some reason
        try:

            print("/"*77)
            print(experiment)
            print("/"*77)
                
            # get the path to all the folder and files
            folders = get_folder(experiment)

            # get the config files
            train_config, test_config = get_configs(folders)

            # get the training data and the model
            if nbr_trials is not None and which_classifier is not None:
                train_loader, model = get_setup(train_config, folders, which_split=train_config["dataset"]["split"], nbr_of_samples_per_class=train_config["dataset"]["nbr_of_samples_per_class"], load_model=True)
            else:
                train_loader, model = get_setup(train_config, folders, which_split=train_config["dataset"]["split"], load_model=True)

            # get eval loader
            test_config["dataset"]["split"] = "test"
            if train_config["dataset"]["name"].lower() == "sviro_uncertainty":
                if "seats" in train_config["dataset"]["split"].lower():
                    eval_loader = get_setup(test_config, folders, which_split="test-adults-and-seats", load_model=False)
                else:
                    eval_loader = get_setup(test_config, folders, which_split="test-adults", load_model=False)
            else:
                eval_loader = get_setup(test_config, folders, which_split="test", load_model=False, nbr_of_samples_per_class=train_config["dataset"]["nbr_of_samples_per_class"])

            analyse_model(model, train_config, train_loader, test_config, eval_loader, folders)
            analyse_model(model, train_config, train_loader, train_config, train_loader, folders)

            test_config["dataset"]["name"] = "orss"
            test_config["dataset"]["split"] = "all"
            # test_config["dataset"]["split"] = "test"
            test_config["dataset"]["factor"] = "X5"
            eval_loader = get_setup(test_config, folders, which_split="all", load_model=False)
            # eval_loader = get_setup(test_config, folders, which_split="test", load_model=False)
            analyse_model(model, train_config, train_loader, test_config, eval_loader, folders)

            test_config["dataset"]["name"] = "orss"
            test_config["dataset"]["split"] = "all"
            # test_config["dataset"]["split"] = "test"
            test_config["dataset"]["factor"] = "Sharan"
            eval_loader = get_setup(test_config, folders, which_split="all", load_model=False)
            # eval_loader = get_setup(test_config, folders, which_split="test", load_model=False)
            analyse_model(model, train_config, train_loader, test_config, eval_loader, folders)


        except KeyboardInterrupt as e:
            print("Interupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"An error occured while processing experiment {experiment}: {e} ")

        # reset the stdout with the original one
        # this is necessary when the train function is called several times
        # by another script
        sys.stdout = sys.stdout.end()


##############################################################################################################################################################

if __name__ == "__main__":

    # specify which gpu should be visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # number of trials per image to do
    # nbr_trials = 20
    # nbr_trials = None

    # which classifier to use
    # which_classifier = "mlp"
    # which_classifier = None

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------

    # list of all experiments to check
    experiments = [

        
    ]

    # evaluate all the experiments from the list
    evaluate_experiments(experiments, nbr_trials, which_classifier)

