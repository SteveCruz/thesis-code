##############################################################################################################################################################
##############################################################################################################################################################
"""
We used this heuristic and scripts to compute the complexity of the different datasets.
If you want to use this script, then change the folder locations of the dataset.
For example, in the following lines:

## Compute TICaM complexity
dataset_dir = "" # location to TICAM train folder"

## Compute MPI3D complexity
dataset_dir = "" # location to real.npz
"""
####################################################################################################################################################
####################################################################################################################################################

import numpy as np
from pathlib import Path
from PIL import Image
from skimage import measure
from skimage.feature import greycomatrix
from tqdm import tqdm

import torch
import dataset

####################################################################################################################################################
####################################################################################################################################################

def compute_complexity_of_dataset(images):

    entropy_values = {
        "shannon": [],
        "glcm": [],
    }

    rgb_weights = [0.2989, 0.5870, 0.1140]

    for image in tqdm(images):


        entropy = measure.shannon_entropy(image)
        entropy_values["shannon"].append(entropy)

        image = np.array(image)

        if image.shape[-1] == 3:
            image = np.dot(image, rgb_weights).astype(np.uint8)
        glcm = np.squeeze(greycomatrix(image, distances=[1], angles=[0], symmetric=True, normed=True))
        entropy = -np.sum(glcm*np.log2(glcm + (glcm==0)))
        entropy_values["glcm"].append(entropy)

    entropy_values = {x:np.array(y) for x,y in entropy_values.items()}  
    mean_entropy = {x:np.mean(y) for x,y in entropy_values.items()}  

    for x,y in mean_entropy.items():
        print(f"Mean {x}: {y}")

####################################################################################################################################################

if __name__ == "__main__":

    ## Compute TICaM complexity
    # dataset_dir = "" # location to TICAM train folder
    # dataset_dir = Path(dataset_dir)
    # image_list = list(dataset_dir.glob("**/*.png"))
    # images = [Image.open(x).convert("L") for x in image_list]
    # compute_complexity_of_dataset(images)

    ## Compute MPI3D complexity
    # dataset_dir = "" # location to real.npz
    # images = np.load(dataset_dir)['images']
    # images = images.reshape(6,6,2,3,3,40,40,64,64,3)
    # images = images[:,:,1,:,:,:,:,:,:,:]
    # images = images.reshape(-1, 64, 64, 3)
    # compute_complexity_of_dataset(images)

    ## Compute GTSRB complexity
    # dataset_dir = "/data/workingdir/sds/data/GTSRB/train_png" # location to TICAM train folder
    # dataset_dir = Path(dataset_dir)
    # image_list = list(dataset_dir.glob("**/*.png"))
    # images = [Image.open(x).convert("L") for x in image_list]
    # compute_complexity_of_dataset(images)

    ## Compute ORSS-Sharan complexity
    # dataset_dir = "/data/workingdir/sds/data/ORSS/Sharan/train_pp_640_64" # location to TICAM train folder
    # dataset_dir = Path(dataset_dir)
    # image_list = list(dataset_dir.glob("*.png"))
    # images = [Image.open(x).convert("L") for x in image_list]
    # compute_complexity_of_dataset(images)

    ## Compute ORSS-X5 complexity
    # dataset_dir = "/data/workingdir/sds/data/ORSS/X5/train_pp_640_64" # location to TICAM train folder
    # dataset_dir = Path(dataset_dir)
    # image_list = list(dataset_dir.glob("*.png"))
    # images = [Image.open(x).convert("L") for x in image_list]
    # compute_complexity_of_dataset(images)

    # Compute SVIRO-Uncertainty complexity
    # dataset_dir = "/data/workingdir/sds/data/SVIRO-Uncertainty/sharan/pp_train-adults-and-seats_64" # location to TICAM train folder
    # dataset_dir = Path(dataset_dir)
    # image_list = list(dataset_dir.glob("**/*.png"))
    # images = [Image.open(x).convert("L") for x in image_list]
    # compute_complexity_of_dataset(images)

    # Compute SVIRO-Illumination complexity
    # dataset_dir = "/data/workingdir/sds/data/SVIRO-Illumination/cayenne/pp_train_128" # location to TICAM train folder
    # dataset_dir = Path(dataset_dir)
    # image_list = list(dataset_dir.glob("**/*.png"))
    # images = [Image.open(x).convert("L") for x in image_list]
    # compute_complexity_of_dataset(images)

    # Compute SVIRO complexity
    # dataset_dir = "/data/workingdir/sds/data/SVIRO/tesla/train/grayscale_wholeImage_pp_640_64" # location to TICAM train folder
    # dataset_dir = Path(dataset_dir)
    # image_list = list(dataset_dir.glob("*.png"))
    # images = [Image.open(x).convert("L") for x in image_list]
    # compute_complexity_of_dataset(images)

    ## Compute MNIST complexity
    # mnist_dataset = dataset.MNIST(which_split="train", make_instance_impossible=False, nbr_of_samples_per_class=-1, augment=False)
    # images = torch.stack(mnist_dataset.images).squeeze().numpy()
    # images = images * 255
    # images = images.astype(np.uint8)
    # compute_complexity_of_dataset(images)

    ## Compute Fashion-MNIST complexity
    # fashion_dataset = dataset.Fashion(which_split="train", make_instance_impossible=False, nbr_of_samples_per_class=-1, augment=False)
    # images = fashion_dataset.data
    # images = [Image.fromarray(x.numpy(), mode='L') for x in images]
    # compute_complexity_of_dataset(images)

    # cifar_dataset = dataset.CIFAR10(which_split="train", make_instance_impossible=False, nbr_of_samples_per_class=-1, augment=False)
    # images = torch.stack(cifar_dataset.images).squeeze().numpy()
    # images = [Image.fromarray(x, mode='L') for x in images]
    # compute_complexity_of_dataset(images)

    # svhn_dataset = dataset.SVHN(which_split="train", make_instance_impossible=False, nbr_of_samples_per_class=-1, augment=False)
    # images = torch.stack(svhn_dataset.images).squeeze().numpy()
    # images = [Image.fromarray(x, mode='L') for x in images]
    # compute_complexity_of_dataset(images)
    
