##############################################################################################################################################################
##############################################################################################################################################################
"""
Automatically remove the background using Otsu's method.
This works best if the II-PIRL and multi-channel autoencoder are used and trained on SVIRO-NoCar.

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
import cv2 as cv
import numpy as np
from pathlib import Path
from importlib import import_module

from torch import nn
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')

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
    folders["images"] = folders["experiment"] / "images" / "remove_bg"
    folders["checkpoint"] = folders["experiment"] / "checkpoints" 
    folders["config"] = folders["experiment"] / "cfg.toml"

    # check if the folder exists
    if not folders["experiment"].is_dir():
        print("The specified experiment does not exist: Exit script.")
        sys.exit()

    folders["images"].mkdir(parents=True, exist_ok=True)
    
    return folders

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

def remove_bg(recon_img, input_img, method="otsu"):
    """
    https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
    https://stackoverflow.com/a/10317883/7042511
    """

    # make sure its a uin8 image
    if recon_img.max() <= 1.0:
        recon_img = recon_img * 255
        recon_img = recon_img.astype(np.uint8)

    # make sure its a 1-channel image, without channel dimension
    if len(recon_img.shape) == 3:
        recon_img = recon_img[0]

    if len(input_img.shape) == 3:
        input_img = input_img[0]

    # keep track of original reconstruction
    org_recon_img = recon_img.copy()

    # apply median blur to recon image
    recon_img = cv.medianBlur(recon_img, 7)

    # apply otsu or triangle thresholding on blurred image
    if method == "otsu":
        treshold_value, masked_recon_img = cv.threshold(recon_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    elif method == "triangle":
        treshold_value, masked_recon_img = cv.threshold(recon_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_TRIANGLE)
    else:
        ValueError("Method is not yet implemented.")

    # try to fill in some simple holes in the mask
    mask = cv.bitwise_not(masked_recon_img)
    contour,_ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv.drawContours(mask,[cnt],0,255,-1)
    mask = cv.bitwise_not(mask)

    # mask the input image using the mask calculated on the reconstruction
    masked_input_img = input_img.copy()
    masked_input_img[mask > treshold_value] = 0
    masked_recon_img = org_recon_img.copy()
    masked_recon_img[mask > treshold_value] = 0

    return org_recon_img, input_img, masked_recon_img, masked_input_img, mask

##############################################################################################################################################################

def remove_background(model, test_loader, config, folders, split):

    # make sure we are in eval mode
    model.eval()

    # we do not need to keep track of gradients
    with torch.no_grad():

        # for each batch of images
        torch.manual_seed("0")
        for idx_batch, batch in enumerate(test_loader):

            # push to gpu
            input_images = batch["image"].to(config["device"])

            if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                input_type = "real" if test_loader.dataset.__class__.__name__.lower() in ["orss", "ticam"] else "synth"
                model_output = model(input_images, input_type=input_type)["xhat"]
            else:
                model_output = model(input_images)["xhat"]

            fig, ax = plt.subplots(5,8, figsize=(8*4, 5*4))
            for ii in range(8):
                one_recon_image = model_output[ii, 0].detach().cpu().numpy()
                one_input_image = input_images[ii, 0].detach().cpu().numpy()
                try:
                    org_recon_img, input_img, masked_recon_img, masked_input_img, mask = remove_bg(one_recon_image, one_input_image)
                except:
                    pass

                # plot the result
                utils.plot_img_to_xis(input_img, ax[0, ii])
                utils.plot_img_to_xis(org_recon_img, ax[1, ii])
                utils.plot_img_to_xis(mask, ax[2, ii])
                utils.plot_img_to_xis(masked_input_img, ax[3, ii])
                utils.plot_img_to_xis(masked_recon_img, ax[4, ii])


            # make sure there is no white space between the subfigures
            plt.subplots_adjust(wspace=0.0, hspace=0.0)
            fig.patch.set_facecolor('black')

            # save the explanation figures, make sure we remove most of the outside white space
            fig_save_path = folders["images"] / f"split_{split}_batch_{idx_batch}.png"
            fig.savefig(fig_save_path, dpi=fig.dpi, bbox_inches='tight', transparent=False)

            # avoid warning of too many figures
            plt.close(fig)

            if idx_batch == 5:
                return

##############################################################################################################################################################

def evaluate_experiment(experiments):

    # for each experiment
    for experiment in experiments:

        print(experiment)
            
        # get the path to all the folder and files
        folders = get_folder(experiment)

        # load the config file
        config = toml.load(folders["config"])
        test_config = toml.load(folders["config"])

        # define the device
        config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        test_config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # get the training data and the model
        train_loader, model = get_setup(config, folders, which_split=config["dataset"]["split"], load_model=True)
        remove_background(model, train_loader, config, folders, "train")

        eval_loader = get_setup(test_config, folders, which_split="test", load_model=False)
        remove_background(model, eval_loader, config, folders, "test")

        test_config["dataset"]["name"] = "orss"
        test_config["dataset"]["split"] = "test"
        test_config["dataset"]["factor"] = "Sharan"
        eval_loader = get_setup(test_config, folders, which_split="test", load_model=False)

        remove_background(model, eval_loader, config, folders, "test-sharan")


##############################################################################################################################################################

if __name__ == "__main__":

    # specify which gpu should be visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # list of all experiments to check
    experiments = [


    ]

    evaluate_experiment(experiments)


   

