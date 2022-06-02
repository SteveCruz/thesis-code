##############################################################################################################################################################
##############################################################################################################################################################
"""
Training scripts for classification models presented in our paper.

Replace or modify the config file in the following part of the code to make changes to train different models.

# load the config file
config = toml.load("cfg/pretrained_classifier.toml")
"""
##############################################################################################################################################################
##############################################################################################################################################################

import os
import sys
import toml
import torch
import random
import numpy as np

from torch import optim
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as TF

import train_ae
import utils as utils
from pretrained_model import PretrainedClassifier, classification_loss
from dataset import create_dataset

##############################################################################################################################################################
##############################################################################################################################################################

def model_setup(config):

    if config["model"]["type"] != "classification":
        raise ValueError("Your config is for an autoencoder model, but this script is for classification models. Please use train_ae.py instead.")

    # load data
    train_loader = create_dataset(
        which_dataset=config["dataset"]["name"], 
        which_factor=config["dataset"]["factor"], 
        img_size=config["dataset"]["img_size"], 
        which_split=config["dataset"]["split"],
        make_scene_impossible=False,
        make_instance_impossible=False,
        augment=config["training"]["augment"],
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        nbr_of_samples_per_class=config["dataset"]["nbr_of_samples_per_class"]
    )

    # number of classification classes
    nbr_classes = len(list(train_loader.dataset.label_str_to_int.keys()))

    # define model
    model = PretrainedClassifier(config["model"], nbr_classes=nbr_classes).to(config["device"])

    # get the optimizer defined in the config file
    # load it from the torch module
    optim_def = getattr(optim, config["training"]["optimizer"])

    # create the optimizer 
    optimizer = dict()
    if config["training"]["optimizer"] == "SGD":
        optimizer["method"] = optim_def(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"], momentum=0.9, nesterov=True)
    else:
        optimizer["method"] = optim_def(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])

    print('=' * 73)
    print(optimizer["method"])
    print('=' * 73)

    return model, optimizer, train_loader

##############################################################################################################################################################

def train_one_epoch(model, optimizer, scaler, train_loader, config, nbr_epoch):

    # make sure we are training
    model.train()

    # placeholder 
    total_loss = 0

    # for each batch
    for batch_images in train_loader:

        # set gradients to zero
        model.zero_grad(set_to_none=True)

        # push to gpu
        input_images = batch_images["image"].to(config["device"])
        labels = batch_images["gt"].to(config["device"])
        
        # inference
        with autocast():
            model_output = model(input_images)

        # classification error
        batch_loss = classification_loss(model_output, labels)               

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(batch_loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer["method"])

        # Updates the scale for next iteration.
        scaler.update()

        # update total loss
        total_loss += batch_loss.item()

    print(f"[Training] \tEpoch: {nbr_epoch+1} Total Loss: {total_loss:.4f}")

    return model

##############################################################################################################################################################

def evaluate(model, train_loader, loader_dict, config, save_folder, nbr_epoch):

    # make sure we are evaluating
    model.eval()

    # we do not need to keep track of gradients
    with torch.no_grad():

        # for the loader of each test vehicle
        for vehicle, loader in loader_dict.items():

            correct = 0
            total = 0

            # for each batch
            for batch_images in loader:
                
                # push to gpu
                input_images = batch_images["image"].to(config["device"])
                labels = batch_images["gt"].to(config["device"])
                
                # we input the distorted input image
                classif_output = model(input_images)

                _, predictions = torch.max(classif_output, 1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            # compute the epoch accuracy
            accuracy = correct / total
            print(f"[Testing] \tEpoch: {nbr_epoch+1}, Vehicle: {vehicle}, Accuracy: {100*accuracy:.2f}% ({correct}/{total})")

##############################################################################################################################################################

def train(config):

    #########################################################
    # GPU
    #########################################################

    # specify which gpu should be visible
    os.environ["CUDA_VISIBLE_DEVICES"] = config["training"]["gpu"]

    # save the gpu settings
    config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # gradscaler to improve speed performance with mixed precision training
    scaler = GradScaler()
    
    #########################################################
    # Setup
    #########################################################

    # create the folders for saving
    save_folder = train_ae.folder_setup(config)

    # create the model, optimizer and data loader
    model, optimizer, train_loader = model_setup(config)

    # get also a test loader for evaluation on unseen dataset
    test_loader = train_ae.get_test_loader(config, get_train_loader=False)

    #########################################################
    # Training
    #########################################################

    # keep track of time
    timer = utils.TrainingTimer()

    # for each epoch
    for nbr_epoch in range(config["training"]["epochs"]):

        # train a single epoch
        model = train_one_epoch(model, optimizer, scaler, train_loader, config, nbr_epoch)

        # evaluate a single epoch
        if (nbr_epoch+1) % config["training"]["frequency"] == 0 or nbr_epoch == 1:
            evaluate(model, train_loader, test_loader, config, save_folder, nbr_epoch)

    #########################################################
    # Aftermath
    #########################################################     

    # save the last model
    torch.save(model.state_dict(), save_folder["checkpoints"] / "last_model.pth")

    print("=" * 37)
    timer.print_end_time()
    print("=" * 37)

    # reset the stdout with the original one
    # this is necessary when the train function is called several times
    # by another script
    sys.stdout = sys.stdout.end()
    
##############################################################################################################################################################
##############################################################################################################################################################

if __name__ == "__main__":

    # reproducibility
    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)
    # random.seed(seed)

    # load the config file
    config = toml.load("config/pretrained_classifier.toml")

    # start the training using the config file
    train(config)