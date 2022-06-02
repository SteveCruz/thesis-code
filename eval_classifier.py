##############################################################################################################################################################
##############################################################################################################################################################
"""
Evaluate classification models after training.

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
import copy
import toml
import torch
import numpy as np
from pathlib import Path
from importlib import import_module

import matplotlib as mpl
from matplotlib import pyplot as plt
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')
# force matplotlib to not plot images
mpl.use("Agg")

import torchvision.transforms.functional as TF

import utils
import dataset

##############################################################################################################################################################
##############################################################################################################################################################

def get_folder(experiment):

    folders = dict()
    folders["experiment"] = Path("results") / experiment
    folders["scripts"] = folders["experiment"] / "scripts"
    folders["logs"] = folders["experiment"] / "logs"
    folders["latent"] = folders["experiment"] / "data" / "latent"
    folders["images"] = folders["experiment"] / "images" / "predictions"
    folders["checkpoint"] = folders["experiment"] / "checkpoints" 
    folders["config"] = folders["experiment"] / "cfg.toml"

    folders["images"].mkdir(parents=True, exist_ok=True)
    
    # save the console output to a file
    sys.stdout = utils.Tee(original_stdout=sys.stdout, file=folders["logs"] / "testing.log")

    return folders

##############################################################################################################################################################        

def get_setup(config, folders, which_split, load_model=False, nbr_of_samples_per_class=-1, print_dataset=False):

    if config["model"]["type"] != "classification":
        raise ValueError("Your config is for a autoencoder model, but this script is for classification models. Please use eval_ae.py instead.")

    # load data
    train_loader = dataset.create_dataset(
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
        model_file = folders["scripts"] / "pretrained_model"

        # replace the / by .
        string_model_file = str(model_file).replace("/",".")

        # load the model definitions from the file
        model_file = import_module(string_model_file)

        # get the autoencoder model definition
        model_def = getattr(model_file, "PretrainedClassifier")

        # number of classification classes
        nbr_classes = len(list(train_loader.dataset.label_str_to_int.keys()))

        # define autoencoder model
        model = model_def(config["model"], nbr_classes=nbr_classes, print_model=False).to(config["device"])

        # load the autoencoder model weights
        checkpoint = torch.load(folders["checkpoint"] / "last_model.pth", map_location=config["device"])

        # apply the model weights
        model.load_state_dict(checkpoint) 

        return train_loader, model

    return train_loader

##############################################################################################################################################################

def evaluate_model(model, train_loader, train_config, test_loader, test_config, folders, split):

    # placeholder
    correct = 0
    total = 0

    # make sure we are in eval mode
    model.eval()

    # we do not need to keep track of gradients
    with torch.no_grad():
        for batch in test_loader:

            images_to_use = []
            labels_to_use = []

            # push to gpu
            input_images = batch["image"].to(train_config["device"])
            labels = batch["gt"].to(train_config["device"])

            for x,y in zip(input_images, labels):
                z = test_loader.dataset.int_to_label_str[int(y.cpu().numpy())]
                # if "2" in z:
                    # continue
                z = [int(z[0]),int(z[2]),int(z[4])]
                if train_config["dataset"]["name"] in ["orss_and_synth_nocar", "orss_and_synth_uncertainty"]:
                    if z in train_loader.dataset.datasets[0].labels:
                        images_to_use.append(x)
                        labels_to_use.append(y)
                else:
                    if z in train_loader.dataset.labels:
                        images_to_use.append(x)
                        labels_to_use.append(y)

            images_to_use = torch.stack(images_to_use)
            labels_to_use = torch.stack(labels_to_use)

            classif_output = model(images_to_use)

            _, predictions = torch.max(classif_output, 1)
            correct += (predictions == labels_to_use).sum().item()
            total += labels_to_use.size(0)

    # compute the epoch accuracy
    accuracy = correct / total

    print(f"[Testing] Vehicle: {split}, Accuracy: {100*accuracy:.2f}% ({correct}/{total})")

    # save accuracy to file
    save_path = folders["logs"] / f"{split}.accuracy"
    utils.save_accuracy(save_path, accuracy)

##############################################################################################################################################################

if __name__ == "__main__":

    # list of all experiments to check
    experiments = [


    ]

    # specify which gpu should be visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # the main folder containing all results
    result_folder = Path("results")

    # for each experiment
    for experiment in experiments:

        print(experiment)
            
        # get the path to all the folder and files
        folders = get_folder(experiment)

        # check if the folder exists
        if not folders["experiment"].is_dir():
            print("The specified experiment does not exist: Exit script.")
            sys.exit()

        # load the config file
        config = toml.load(folders["config"])

        # define the device
        config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # get the training data and the model
        train_loader, model = get_setup(config, folders, which_split=config["dataset"]["split"], nbr_of_samples_per_class=config["dataset"]["nbr_of_samples_per_class"], load_model=True)

        vehicle_config = copy.deepcopy(config)
        vehicle_config["dataset"]["name"] = "orss"
        vehicle_config["dataset"]["split"] = "all"
        # vehicle_config["dataset"]["split"] = "test"
        vehicle_config["dataset"]["factor"] = "X5"

        vehicle_eval_loader = get_setup(vehicle_config, folders, which_split="all", load_model=False)
        evaluate_model(model, train_loader, config, vehicle_eval_loader, vehicle_config, folders, split="orss-all-x5")
        # vehicle_eval_loader = get_setup(vehicle_config, folders, which_split="test", load_model=False)
        # evaluate_model(model, train_loader, config, vehicle_eval_loader, vehicle_config, folders, split="orss-test-x5")
        
        vehicle_config = copy.deepcopy(config)
        vehicle_config["dataset"]["name"] = "orss"
        vehicle_config["dataset"]["split"] = "all"
        # vehicle_config["dataset"]["split"] = "test"
        vehicle_config["dataset"]["factor"] = "Sharan"

        vehicle_eval_loader = get_setup(vehicle_config, folders, which_split="all", load_model=False)
        evaluate_model(model, train_loader, config, vehicle_eval_loader, vehicle_config, folders, split="orss-all-sharan")
        # vehicle_eval_loader = get_setup(vehicle_config, folders, which_split="test", load_model=False)
        # evaluate_model(model, train_loader, config, vehicle_eval_loader, vehicle_config, folders, split="orss-test-sharan")

        # reset the stdout with the original one
        # this is necessary when the train function is called several times
        # by another script
        sys.stdout = sys.stdout.end()
