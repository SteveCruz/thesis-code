##############################################################################################################################################################
##############################################################################################################################################################
"""
Training scripts for autoencoder models presented in our paper.

Replace or modify the config file in the following part of the code to make changes to train different models.

# load the config file
config = toml.load("cfg/conv_ae.toml")
"""
##############################################################################################################################################################
##############################################################################################################################################################

import os
import sys
import time
import toml
import torch
import random

import numpy as np
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from shutil import copyfile

import utils as utils
from model import create_ae_model
from dataset import create_dataset

##############################################################################################################################################################
##############################################################################################################################################################

def folder_setup(config):

    # define the name of the experiment
    if config["model"]["type"] == "classification":
        experiment_name = time.strftime("%Y%m%d-%H%M%S") + "_" + config["model"]["type"] + "_" + config["dataset"]["name"]
    elif config["model"]["type"] in ["mlp", "conv"]:
        experiment_name = time.strftime("%Y%m%d-%H%M%S") + "_" + config["model"]["type"] + "_" + config["dataset"]["name"]
    else:
        experiment_name = time.strftime("%Y%m%d-%H%M%S") + "_" + config["model"]["type"] + "_" + config["dataset"]["name"]  + "_latentDimension_" + str(config["model"]["dimension"])

    # define the paths to save the experiment
    save_folder = dict()
    Path("results").mkdir(exist_ok=True)
    save_folder["main"] = Path("results") / experiment_name
    save_folder["checkpoints"] = save_folder["main"] / "checkpoints" 
    save_folder["images"] = save_folder["main"] / "images" / "train"
    save_folder["data"] = save_folder["main"] / "data"
    save_folder["latent"] = save_folder["main"] / "data" / "latent"
    save_folder["scripts"] = save_folder["main"] / "scripts"
    save_folder["logs"] = save_folder["main"] / "logs"

    # create all the folders
    for item in save_folder.values():
        item.mkdir(parents=True)

    # save the console output to a file and to the console
    sys.stdout = utils.Tee(original_stdout=sys.stdout, file=save_folder["logs"] / "training.log")

    # save the accuracy for each evaluation to a file
    save_folder["accuracy"] = save_folder["logs"] / "accuracy.log"
    save_folder["accuracy"].touch()

    # copy files as a version backup
    # this way we know exactly what we did
    # these can also be loaded automatically for testing the models
    copyfile(Path(__file__).absolute(), save_folder["scripts"] / "train_ae.py")
    copyfile(Path().absolute() / "dataset.py", save_folder["scripts"] / "dataset.py")
    copyfile(Path().absolute() / "model.py", save_folder["scripts"] / "model.py")
    copyfile(Path().absolute() / "utils.py", save_folder["scripts"] / "utils.py")
    copyfile(Path().absolute() / "pretrained_model.py", save_folder["scripts"] / "pretrained_model.py")

    # save config file
    # remove device info, as it is not properly saved
    config_to_save = config.copy()
    del config_to_save["device"]
    utils.write_toml_to_file(config_to_save, save_folder["main"] / "cfg.toml")

    return save_folder

##############################################################################################################################################################

def model_setup(config):

    # define model and print it
    model = create_ae_model(config).to(config["device"])
    model.print_model()

    # get the optimizer defined in the config file
    # load it from the torch module
    optim_def = getattr(optim, config["training"]["optimizer"])

    # create the optimizer 
    if config["training"]["optimizer"] == "SGD":
        optimizer = optim_def(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"], momentum=0.9, nesterov=True)
    else:
        optimizer = optim_def(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])

    print('=' * 73)
    print(optimizer)
    print('=' * 73)

    # load data
    train_loader = create_dataset(
        which_dataset=config["dataset"]["name"], 
        which_factor=config["dataset"]["factor"], 
        img_size=config["dataset"]["img_size"], 
        which_split=config["dataset"]["split"],
        make_scene_impossible=config["training"]["make_scene_impossible"],
        make_instance_impossible=config["training"]["make_instance_impossible"],
        augment=config["training"]["augment"],
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        nbr_of_samples_per_class=config["dataset"]["nbr_of_samples_per_class"]
    )

    return model, optimizer, train_loader

##############################################################################################################################################################

def get_test_loader(config, get_train_loader=False):

    # dict to keep a loader for each test vehicle
    test_loader = dict()

    if "sviro" not in config["dataset"]["name"].lower():
        test_loader["test"] = create_dataset(
            which_dataset=config["dataset"]["name"], 
            which_factor=config["dataset"]["factor"], 
            img_size=config["dataset"]["img_size"], 
            which_split="test",
            batch_size=config["training"]["batch_size"],
            make_scene_impossible=False,
            make_instance_impossible=False,
            shuffle=True,
        )

    elif config["dataset"]["name"].lower() == "sviro_uncertainty":

        test_loader["test-adults"] = create_dataset(
            which_dataset=config["dataset"]["name"], 
            which_factor=config["dataset"]["factor"], 
            img_size=config["dataset"]["img_size"], 
            which_split="test-adults",
            batch_size=config["training"]["batch_size"],
            make_scene_impossible=False,
            make_instance_impossible=False,
            shuffle=True,
        )

        test_loader["test-objects"] = create_dataset(
            which_dataset=config["dataset"]["name"], 
            which_factor=config["dataset"]["factor"], 
            img_size=config["dataset"]["img_size"], 
            which_split="test-objects",
            batch_size=config["training"]["batch_size"],
            make_scene_impossible=False,
            make_instance_impossible=False,
            shuffle=True,
        )

        test_loader["test-adults-and-objects"] = create_dataset(
            which_dataset=config["dataset"]["name"], 
            which_factor=config["dataset"]["factor"], 
            img_size=config["dataset"]["img_size"], 
            which_split="test-adults-and-objects",
            batch_size=config["training"]["batch_size"],
            make_scene_impossible=False,
            make_instance_impossible=False,
            shuffle=True,
        )

        test_loader["test-adults-and-seats"] = create_dataset(
            which_dataset=config["dataset"]["name"], 
            which_factor=config["dataset"]["factor"], 
            img_size=config["dataset"]["img_size"], 
            which_split="test-adults-and-seats",
            batch_size=config["training"]["batch_size"],
            make_scene_impossible=False,
            make_instance_impossible=False,
            shuffle=True,
        )

    else:

        test_loader["test"] = create_dataset(
            which_dataset=config["dataset"]["name"], 
            which_factor=config["dataset"]["factor"], 
            img_size=config["dataset"]["img_size"], 
            which_split="test",
            batch_size=config["training"]["batch_size"],
            make_scene_impossible=False,
            make_instance_impossible=False,
            shuffle=True,
        )

    if get_train_loader:
        test_loader["train"] = create_dataset(
            which_dataset=config["dataset"]["name"], 
            which_factor=config["dataset"]["factor"], 
            img_size=config["dataset"]["img_size"], 
            which_split=config["dataset"]["split"],
            batch_size=config["training"]["batch_size"],
            make_scene_impossible=False,
            make_instance_impossible=False,
            shuffle=True, 
        )

    return test_loader

##############################################################################################################################################################

def train_one_epoch(model, optimizer, scaler, train_loader, config, save_folder, nbr_epoch):

    # make sure we are training
    model.train()

    # init
    total_loss = 0
    total_recon_loss = 0
    total_metric_loss = 0
    total_kl_loss = 0

    # for each batch
    for batch_idx, batch_images in enumerate(train_loader):

        # init
        batch_loss = 0
        batch_recon_loss = 0
        batch_metric_loss = 0
        batch_kl_loss = 0
        
        # set gradients to zero
        model.zero_grad(set_to_none=True)

        # push to gpu
        input_images = batch_images["image"].to(config["device"])
        target_images = batch_images["target"].to(config["device"])

        with autocast():
            # inference
            model_output = model(input_images)

        # reconstruction error
        batch_recon_loss = model.loss(prediction=model_output["xhat"].to(torch.float32), target=target_images)
        batch_loss += batch_recon_loss 

        # metric loss
        if config["model"]["metric"]:
            batch_metric_loss = model.metric_loss(embedding=model_output["mu"], labels=batch_images["gt"])
            batch_loss += batch_metric_loss

        if model.type in ["conv_vae", "fc_vae"]:
            batch_kl_loss = model.kl_divergence_loss(model_output["mu"], model_output["logvar"])
            batch_loss += config["training"]["kl_weight"]*batch_kl_loss

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(batch_loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # accumulate loss
        total_loss += batch_loss.item()
        total_recon_loss += batch_recon_loss.item()
        if config["model"]["metric"]: total_metric_loss += batch_metric_loss.item()
        if model.type in ["conv_vae", "fc_vae"]: total_kl_loss += batch_kl_loss.item()

        # plot the result sometimes
        if ((nbr_epoch+1) % config["training"]["frequency"] == 0 or nbr_epoch == 1) and batch_idx == 0:
            utils.plot_progress(images=[input_images, model_output["xhat"], target_images], save_folder=save_folder, nbr_epoch=nbr_epoch, text="epoch")

    if (nbr_epoch+1) % 100 == 0:  
        if model.type in ["conv_vae", "fc_vae"]:
            print(f"[Training] \tEpoch: {nbr_epoch+1} Total Loss: {total_loss:.2f} \tRecon Loss: {total_recon_loss:.2f} \tMetric Loss - {config['model']['metric']}: {total_metric_loss:.2f} \tKL Loss: {total_kl_loss:.2f}")
        elif config["model"]["metric"]:
            print(f"[Training] \tEpoch: {nbr_epoch+1} Total Loss: {total_loss:.2f} \tRecon Loss: {total_recon_loss:.2f} \tMetric Loss - {config['model']['metric']}: {total_metric_loss:.2f}")
        else:
            print(f"[Training] \tEpoch: {nbr_epoch+1} Total Loss: {total_loss:.2f} \tRecon Loss: {total_recon_loss:.2f}")

    return model

##############################################################################################################################################################

def train_one_epoch_multichannel(model, optimizer, scaler, train_loader, config, save_folder, nbr_epoch):

    # make sure we are training
    model.train()

    # init
    total_loss = 0
    total_synthetic_recon_loss = 0
    total_real_recon_loss = 0
    total_metric_loss = 0

    # for each batch
    for batch_idx, batch_images in enumerate(train_loader):

        # init
        batch_loss = 0
        batch_synthetic_recon_loss = 0
        batch_real_recon_loss = 0
        batch_metric_loss = 0
        batch_synthetic_loss = 0
        batch_real_loss = 0
        
        # set gradients to zero
        model.zero_grad(set_to_none=True)

        # push to gpu
        synthetic_input_images = batch_images["image"].to(config["device"])
        synthetic_target_images = batch_images["target"].to(config["device"])
        # synthetic_target_images = batch_images["real_target"].to(config["device"])
        real_input_images = batch_images["real_image"].to(config["device"])
        # real_target_images = batch_images["real_target"].to(config["device"])
        real_target_images = batch_images["target"].to(config["device"])

        # inference
        with autocast():
            synthetic_model_output = model(synthetic_input_images, input_type="synth")
            real_model_output = model(real_input_images, input_type="real")

        # reconstruction error
        batch_synthetic_recon_loss = model.loss(prediction=synthetic_model_output["xhat"].to(torch.float32), target=synthetic_target_images)
        batch_synthetic_loss += batch_synthetic_recon_loss 

        batch_real_recon_loss = model.loss(prediction=real_model_output["xhat"].to(torch.float32), target=real_target_images)
        batch_real_loss += batch_real_recon_loss 

        # metric loss
        if config["model"]["metric"]:
            tmp1 = torch.cat([synthetic_model_output["mu"], real_model_output["mu"]])
            tmp2 = torch.cat([batch_images["gt"], batch_images["gt"]])
            batch_metric_loss = model.metric_loss(embedding=tmp1, labels=tmp2)
            
        # multi-channel loss
        batch_loss = batch_synthetic_loss + batch_real_loss + batch_metric_loss

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(batch_loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # accumulate loss
        total_loss += batch_loss.item()
        total_synthetic_recon_loss += batch_synthetic_recon_loss.item()
        total_real_recon_loss += batch_real_recon_loss.item()
        if config["model"]["metric"]: total_metric_loss += batch_metric_loss.item()

        # plot the result sometimes
        if ((nbr_epoch+1) % config["training"]["frequency"] == 0 or nbr_epoch == 1) and batch_idx == 0:
            utils.plot_progress(images=[synthetic_input_images, synthetic_model_output["xhat"], synthetic_target_images], save_folder=save_folder, nbr_epoch=nbr_epoch, text="epoch_synthetic")
            utils.plot_progress(images=[real_input_images, real_model_output["xhat"], real_target_images], save_folder=save_folder, nbr_epoch=nbr_epoch, text="epoch_real")


    if (nbr_epoch+1) % 100 == 0:  
        if config["model"]["metric"]:
            print(f"[Training] \tEpoch: {nbr_epoch+1} Total Loss: {total_loss:.2f} \tRecon Synth Loss: {total_synthetic_recon_loss:.2f} \tRecon Real Loss: {total_real_recon_loss:.2f} \tMetric Loss - {config['model']['metric']}: {total_metric_loss:.2f}")
        else:
            print(f"[Training] \tEpoch: {nbr_epoch+1} Total Loss: {total_loss:.2f} \tRecon Synth Loss: {total_synthetic_recon_loss:.2f} \tRecon Real Loss: {total_real_recon_loss:.2f}")

    return model

##############################################################################################################################################################

def train_one_epoch_classifier(model, optimizer, scaler, train_loader, config, nbr_epoch):

    # make sure we are training
    model.train()

    # init
    total_loss = 0

    # for each batch
    for batch_images in train_loader:

        # init
        batch_loss = 0
        
        # set gradients to zero
        model.zero_grad(set_to_none=True)

        # push to gpu
        input_images = batch_images["image"].to(config["device"])
        gts = batch_images["gt"].to(config["device"])

        with autocast():
            # inference
            model_output = model(input_images)

            # reconstruction error
            batch_recon_loss = model.loss(prediction=model_output["prediction"], target=gts)
            batch_loss += batch_recon_loss 

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(batch_loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.g
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # accumulate loss
        total_loss += batch_loss.item()

    if (nbr_epoch+1) % 100 == 0:  
        print(f"[Training] \tEpoch: {nbr_epoch+1} Total Loss: {total_loss:.6f} ")

    return model

##############################################################################################################################################################

def recon_one_batch(model, loader_dict, config, save_folder, nbr_epoch):

    if (nbr_epoch+1) % config["training"]["frequency"] == 0 or nbr_epoch == 1:

        # save the current model state
        torch.save(model.state_dict(), save_folder["checkpoints"] / "last_model.pth")

        # make sure we are in eval mode
        model.eval()

        # do not keep track of gradients
        with torch.no_grad():

            # for the loader of each test vehicle
            for vehicle, loader in loader_dict.items():
                
                # for each batch of training images
                for batch_idx, input_images in enumerate(loader):

                    input_images = input_images["image"].to(config["device"])
                    
                    if model.type in ["multi_channel_ae", "multi_channel_extractor_ae"] :
                        input_type = "real" if vehicle.lower() in ["x5", "sharan", "ticam"] else "synth"
                        model_output = model(input_images, input_type=input_type)
                    else:
                        model_output = model(input_images)

                    # plot the result
                    utils.plot_progress(images=[input_images, model_output["xhat"]], save_folder=save_folder, nbr_epoch=nbr_epoch, text=f"{vehicle}_{batch_idx}_epoch")

                    break

##############################################################################################################################################################

def accuracy_one_epoch(model, loader_dict, config, save_folder, nbr_epoch):

    if (nbr_epoch+1) % config["training"]["frequency"] == 0 or nbr_epoch == 1:

        # save the current model state
        torch.save(model.state_dict(), save_folder["checkpoints"] / "last_model.pth")

        # make sure we are in eval mode
        model.eval()

        # do not keep track of gradients
        with torch.no_grad():

            # for the loader of each test vehicle
            for vehicle, loader in loader_dict.items():

                # keep track
                correct = 0
                total = 0
                
                # for each batch of training images
                for batch in loader:
                    
                    # push to gpu
                    input_images = batch["image"].to(config["device"])
                    gts = batch["gt"].numpy()

                    # encode the images
                    model_output = model(input_images)

                    # performance
                    prediction = model_output["prediction"].cpu().numpy().argmax(axis=1)
                    correct += (prediction == gts).sum()
                    total += prediction.shape[0]

                accuracy = 100 * correct / total
                print(f"Accuracy {vehicle}: {accuracy:.2f}%")

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
    save_folder = folder_setup(config)

    # create the model, optimizer and data loader
    model, optimizer, train_loader = model_setup(config)

    # get also a test loader for evaluation on unseen dataset
    test_loader = get_test_loader(config, get_train_loader=False)

    #########################################################
    # Training
    #########################################################

    # keep track of time
    timer = utils.TrainingTimer()

    # for each epoch
    for nbr_epoch in range(config["training"]["epochs"]):

        if model.type in ["mlp", "conv"]:
            model = train_one_epoch_classifier(model, optimizer, scaler, train_loader, config, nbr_epoch)
            accuracy_one_epoch(model, test_loader, config, save_folder, nbr_epoch)
        elif model.type in ["multi_channel_ae", "multi_channel_extractor_ae"]:
            model = train_one_epoch_multichannel(model, optimizer, scaler, train_loader, config, save_folder, nbr_epoch)
            recon_one_batch(model, test_loader, config, save_folder, nbr_epoch)
        else:
            model = train_one_epoch(model, optimizer, scaler, train_loader, config, save_folder, nbr_epoch)
            recon_one_batch(model, test_loader, config, save_folder, nbr_epoch)

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
    # config = toml.load("config/fc_ae.toml")
    # config = toml.load("config/conv_ae.toml")
    # config = toml.load("config/conv_vae.toml")
    # config = toml.load("config/dropout_fc_ae.toml")
    # config = toml.load("config/dropout_conv_ae.toml")
    # config = toml.load("config/mlp.toml")
    # config = toml.load("config/conv.toml")
    # config = toml.load("config/extractor_ae.toml")
    # config = toml.load("config/multi_channel_ae.toml")
    # config = toml.load("config/multi_channel_extractor_ae.toml")
    
    # start the training using the config file
    train(config)