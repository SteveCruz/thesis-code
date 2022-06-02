####################################################################################################################################################
####################################################################################################################################################
"""
Dataloader definitions for all the datasets used in our paper.
The datasets need to be downloaded manually and placed inside a same folder.
Specify your folder location in the following line

# directory containing all the datasets
ROOT_DATA_DIR = Path("")
"""
####################################################################################################################################################
####################################################################################################################################################

import os
import cv2
import random
import numpy as np
from PIL import Image
from pathlib import Path
from skimage import exposure

import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms.functional as TF
from torchvision.datasets import FashionMNIST as TFashionMNIST
from torchvision.datasets import CIFAR10 as TCIFAR10
from torchvision.datasets import SVHN as TSVHN
from torchvision.datasets import Omniglot as TOmniglot
from torchvision.datasets import Places365 as TPlaces365
from torchvision.datasets import LSUN as TLSUN
from torchvision.datasets import MNIST as TMNIST

import albumentations as album

from collections import defaultdict

####################################################################################################################################################
####################################################################################################################################################

# directory containing all the datasets
ROOT_DATA_DIR = Path("")

####################################################################################################################################################

class BaseDatasetCar(Dataset):
    """
    Base class for all dataset classes for the vehicle interior.
    """

    def __init__(self, root_dir, car, img_size, split, make_scene_impossible, make_instance_impossible, augment=False, nbr_of_samples_per_class=-1):

        # path to the main folder
        self.root_dir =  Path(root_dir)

        # which car are we using?
        self.car = car

        # train or test split
        self.img_size = img_size
        self.split = split

        # are we using training data
        self.is_train = True if "train" in self.split or self.split=="all" else False

        # normal or impossible reconstruction loss?
        self.make_scene_impossible = make_scene_impossible
        self.make_instance_impossible = make_instance_impossible
        if self.make_instance_impossible and self.make_scene_impossible:
            raise ValueError("Cannot have both to be true.")

        # pre-process the data if necessary
        self._pre_process_dataset()

        # load the data into the memory
        self._get_data()

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=3),
                    album.RandomBrightnessContrast(always_apply=False, p=0.4, brightness_limit=(0.0, 0.33), contrast_limit=(0.0, 0.33), brightness_by_max=False),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.10)),
                    # album.CoarseDropout(always_apply=False, p=0.4, fill_value=0.5, min_holes=5, max_holes=5, min_height=12, max_height=12, min_width=12, max_width=12),
                ]
            )
        else:
            self.augment = False

        # dict to match the concatenations of the three seat position classes into a single integer
        self.label_str_to_int = {
            '0_0_0': 0, '0_0_3': 1, '0_3_0': 2, '3_0_0': 3, '0_3_3': 4, '3_0_3': 5, '3_3_0': 6, '3_3_3': 7, 
            '0_0_1': 8, '0_0_2': 9, '0_1_0': 10, '0_1_1': 11, '0_1_2': 12, '0_1_3': 13, '0_2_0': 14, '0_2_1': 15, 
            '0_2_2': 16, '0_2_3': 17, '0_3_1': 18, '0_3_2': 19, '1_0_0': 20, '1_0_1': 21, '1_0_2': 22, '1_0_3': 23, 
            '1_1_0': 24, '1_1_1': 25, '1_1_2': 26, '1_1_3': 27, '1_2_0': 28, '1_2_1': 29, '1_2_2': 30, '1_2_3': 31, 
            '1_3_0': 32, '1_3_1': 33, '1_3_2': 34, '1_3_3': 35, '2_0_0': 36, '2_0_1': 37, '2_0_2': 38, '2_0_3': 39, 
            '2_1_0': 40, '2_1_1': 41, '2_1_2': 42, '2_1_3': 43, '2_2_0': 44, '2_2_1': 45, '2_2_2': 46, '2_2_3': 47, 
            '2_3_0': 48, '2_3_1': 49, '2_3_2': 50, '2_3_3': 51, '3_0_1': 52, '3_0_2': 53, '3_1_0': 54, '3_1_1': 55, 
            '3_1_2': 56, '3_1_3': 57, '3_2_0': 58, '3_2_1': 59, '3_2_2': 60, '3_2_3': 61, '3_3_1': 62, '3_3_2': 63
        }

        # the revers of the above, to transform int labels back into strings
        self.int_to_label_str = {v:k for k,v in self.label_str_to_int.items()}

    def _get_subset_of_data(self):

        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, label in enumerate(self.labels):
                
                # make sure its a string
                label =  self._get_classif_str(label)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
            
            # only take the subset of indices based on how many samples per class to keep
            self.images = [x for idx, x in enumerate(self.images) if idx in keep_indices]
            self.labels = [x for idx, x in enumerate(self.labels) if idx in keep_indices]

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        # number of images to use
        return len(self.images)

    def _get_data(self):

        # get all folders with the sceneries
        if self.split == "all":
            if self.car.lower() == "all":
                self.folders = sorted(list(self.root_dir.glob(f"*/pp_*_{self.img_size}/*")))
            else:
                self.folders = sorted(list(self.root_dir.glob(f"{self.car}/pp_*_{self.img_size}/*")))
        else:
            if self.car.lower() == "all":
                self.folders = sorted(list(self.root_dir.glob(f"*/pp_{self.split}_{self.img_size}/*")))
            else:
                self.folders = sorted(list(self.root_dir.glob(f"{self.car}/pp_{self.split}_{self.img_size}/*")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []
        counter = 0

        # for each folder
        for folder in self.folders:

            # get classification labels for each seat from folder name
            classif_labels = self._get_classif_label(folder)

            # each scene will be an array of images
            self.images.append([])

            # get all the images for this scene
            files = sorted(list(folder.glob("*.png")))

            # for each file
            for file in files:
        
                # open the image specified by the path
                # make sure it is a grayscale image
                img = np.array(Image.open(file).convert("L"))

                # append the image to the placeholder
                self.images[counter].append(img)

            counter += 1
            
            # append label to placeholder
            self.labels.append(classif_labels)

    def _get_classif_label(self, file_path):

        # get the filename only of the path
        name = file_path.stem

        # split at GT 
        gts = name.split("GT")[-1]
        
        # split the elements at _
        # first element is empty string, remove it
        clean_gts = gts.split("_")[1:]

        # convert the strings to ints
        clean_gts = [int(x) for x in clean_gts]

        # convert sviro labels to compare with other datasets
        for index, value in enumerate(clean_gts):
            # everyday objects to background
            if value in [4]:
                clean_gts[index] = 0

        return clean_gts

    def _get_classif_str(self, label):
        return str(label[0]) + "_" + str(label[1]) + "_" + str(label[2])

    def _pre_process_dataset(self):

        # get all the subfolders inside the dataset folder
        data_folder_variations = self.root_dir.glob("*")

        # for each variation
        for folder in data_folder_variations:

            # for each split
            for pre_processed_split in [f"pp_train_{self.img_size}", f"pp_test_{self.img_size}"]:

                # create the path
                path_to_preprocessed_split = folder / pre_processed_split
                path_to_vanilla_split = folder / pre_processed_split.split("_")[1]

                # if no pre-processing for these settings exists, then create them
                if not path_to_preprocessed_split.exists():

                    print("-" * 37)
                    print(f"Pre-process and save data for folder: {folder} and split: {pre_processed_split} and downscale size: {self.img_size} ...")
                    
                    self.pre_process_and_save_data(path_to_preprocessed_split, path_to_vanilla_split)
                    
                    print("Pre-processing and saving finished.")
                    print("-" * 37)

    def pre_process_and_save_data(self, path_to_preprocessed_split, path_to_vanilla_split):
        """
        To speed up training, it is beneficial to do the rudementary pre-processing once and save the data.
        """

        # create the folders to save the pre-processed data
        path_to_preprocessed_split.mkdir()

        # get all the files in all the subfolders 
        files = list(path_to_vanilla_split.glob(f"**/*.png"))

        # for each image 
        for curr_file in files:

            # open the image specified by the path
            img = Image.open(curr_file).convert("L")

            # center crop the image using the smaller size (i.e. width or height)
            # to define the new size of the image (basically we remove only either the width or height)
            img = TF.center_crop(img, np.min(img.size))

            # then resize the image to the one we want to use for training
            img = TF.resize(img, self.img_size)

            # create the folder for the experiment
            save_folder = path_to_preprocessed_split / curr_file.parent.stem
            save_folder.mkdir(exist_ok=True)

            # save the processed image
            img.save(save_folder / curr_file.name)

    def _get_positive(self, rand_indices, positive_label, positive_images):

        # get all the potential candidates which have the same label 
        masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # if there is no other image with the same label
        if not masked:
            
            new_rand_indices = random.sample(range(0,len(positive_images)), 2)
            positive_input_image = positive_images[new_rand_indices[0]]
            positive_output_image = positive_images[new_rand_indices[1]] if self.make_scene_impossible else positive_images[new_rand_indices[0]]
            positive_input_image = TF.to_tensor(positive_input_image)
            positive_output_image = TF.to_tensor(positive_output_image)

        else:
            # choose one index randomly from the masked subset
            index = np.random.choice(masked)

            positive_input_image = self.images[index][rand_indices[0]]
            positive_output_image = self.images[index][rand_indices[1]] if self.make_scene_impossible else self.images[index][rand_indices[0]]
            positive_input_image = TF.to_tensor(positive_input_image)
            positive_output_image = TF.to_tensor(positive_output_image)

        return positive_input_image, positive_output_image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        images = self.images[index]
        label = self.labels[index]

        # randomly selected
        # .) the input images 
        # .) the output images 
        rand_indices = random.sample(range(0,len(images)), 2)

        # get the image to be used as input
        input_image = images[rand_indices[0]]

        # get the image to be used for the reconstruction error
        output_image = images[rand_indices[1]] if self.make_scene_impossible else images[rand_indices[0]]

        # make sure its a tensor
        input_image = TF.to_tensor(input_image)
        output_image = TF.to_tensor(output_image)

        # randomly flip
        if random.choice([True, False]):
            input_image = TF.hflip(input_image)
            output_image = TF.hflip(output_image)
            label = label[::-1]

        str_label = self._get_classif_str(label)

        if self.make_instance_impossible:
            _, output_image = self._get_positive(rand_indices, label, images)

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: input_image = torch.from_numpy(self.augment(image=np.array(input_image)[0])["image"][None,:])

        return {"image":input_image, "target":output_image, "gt": self.label_str_to_int[str_label]}
            
####################################################################################################################################################

class SVIRO(BaseDatasetCar):
    """
    https://sviro.kl.dfki.de

    You only need the grayscale images for the whole scene.
    Make sure to have a folder structure as follows:
    
    SVIRO 
    ├── aclass
    │   ├── train
    │   │   └──── grayscale_wholeImage
    │   └── test
    │       └──── grayscale_wholeImage
    ⋮
    ⋮
    ⋮
    └── zoe
        ├── train
        │   └──── grayscale_wholeImage
        └── test
            └──── grayscale_wholeImage
    """
    def __init__(self, car, img_size, which_split, make_instance_impossible, augment):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "SVIRO"  

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car=car, img_size=img_size, split=which_split, make_scene_impossible=False, make_instance_impossible=make_instance_impossible, augment=augment)
        
    def _get_data(self):

        # get all the png files, i.e. experiments
         # get all folders with the sceneries
        if self.split == "all":
            if self.car.lower() == "all":
                self.files = sorted(list(self.root_dir.glob(f"*/*/grayscale_wholeImage_pp_640_{self.img_size}/*.png")))
            else:
                self.files = sorted(list(self.root_dir.glob(f"{self.car}/*/grayscale_wholeImage_pp_640_{self.img_size}/*.png")))
        else:
            if self.car.lower() == "all":
                self.files = sorted(list(self.root_dir.glob(f"*/{self.split}/grayscale_wholeImage_pp_640_{self.img_size}/*.png")))
            else:
                self.files = sorted(list(self.root_dir.glob(f"{self.car}/{self.split}/grayscale_wholeImage_pp_640_{self.img_size}/*.png")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []

        # for each file
        for file in self.files:
        
            # get classification labels for each seat from folder name
            classif_labels = self._get_classif_label(file)

            # open the image specified by the path
            # make sure it is a grayscale image
            img = np.array(Image.open(file).convert("L"))

            # each scene will be an array of images
            # append the image to the placeholder
            self.images.append([img])
        
            # append label to placeholder
            self.labels.append(classif_labels)

    def _get_classif_label(self, file_path):

        # get the filename only of the path
        name = file_path.stem

        # split at GT 
        gts = name.split("GT")[-1]
        
        # split the elements at _
        # first element is empty string, remove it
        clean_gts = gts.split("_")[1:]

        # convert the strings to ints
        clean_gts = [int(x) for x in clean_gts]

        # convert sviro labels to compare with other datasets
        for index, value in enumerate(clean_gts):
            # everyday objects to background
            if value in [4,5,6]:
                clean_gts[index] = 0

        return clean_gts

    def _pre_process_dataset(self):

        # get all the subfolders inside the dataset folder
        data_folder_variations = self.root_dir.glob("*/*")

        # for each variation
        for folder in data_folder_variations:

            # create the path
            path_to_preprocessed_split = folder / f"grayscale_wholeImage_pp_640_{self.img_size}"
            path_to_vanilla_split = folder / "grayscale_wholeImage"

            # if no pre-processing for these settings exists, then create them
            if not path_to_preprocessed_split.exists():

                print("-" * 37)
                print(f"Pre-process and save data for folder: {folder} and downscale size: {self.img_size} ...")
                
                self.pre_process_and_save_data(path_to_preprocessed_split, path_to_vanilla_split)
                
                print("Pre-processing and saving finished.")
                print("-" * 37)

    def pre_process_and_save_data(self, path_to_preprocessed_split, path_to_vanilla_split):
        """
        To speed up training, it is beneficial to do the rudementary pre-processing once and save the data.
        """

        # create the folders to save the pre-processed data
        path_to_preprocessed_split.mkdir()

        # get all the files in all the subfolders 
        files = list(path_to_vanilla_split.glob("*.png"))

        # for each image 
        for curr_file in files:

            # open the image specified by the path
            img = Image.open(curr_file).convert("L")

            # center crop the image using the smaller size (i.e. width or height)
            # to define the new size of the image (basically we remove only either the width or height)
            img = TF.center_crop(img, np.min(img.size))

            # then resize the image to the one we want to use for training
            img = TF.resize(img, self.img_size)

            # create the path to the file
            save_path = path_to_preprocessed_split / curr_file.name

            # save the processed image
            img.save(save_path)

    def _get_positive(self, positive_label):

        # get all the potential candidates from the real images which have the same label as the synthetic one
        masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        input_image = self.images[index][0]
        input_image = TF.to_tensor(input_image)

        return input_image


    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index][0]
        label = self.labels[index]
        str_label = self._get_classif_str(label)

        # transform it for pytorch (normalized and transposed)
        image = TF.to_tensor(image)
    
        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt": self.label_str_to_int[str_label]}

####################################################################################################################################################

class Seats_And_People_Nocar(BaseDatasetCar):
    """
    Download dataset from the anonymous link provided in the paper or in the README.md.

    Make sure to have a folder structure as follows:

    SEATS_AND_PEOPLE_NOCAR 
    └── cayenne
        ├── pp_train_64
        └── pp_test_64
    """
    

    def __init__(self, car, img_size, which_split, make_scene_impossible, make_instance_impossible, augment):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "SVIRO-NoCar"

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car=car, img_size=img_size, split=which_split, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, augment=augment)
        
####################################################################################################################################################

class SVIROIllumination(BaseDatasetCar):
    """
    https://sviro.kl.dfki.de
    
    Make sure to have a folder structure as follows:

    SVIRO-Illumination 
    ├── cayenne
    │   ├── train
    │   └── test
    ├── kodiaq
    │   ├── train
    │   └── test
    └── kona
        ├── train
        └── test
    """
    def __init__(self, car, img_size, which_split, make_scene_impossible, make_instance_impossible, augment):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "SVIRO-Illumination"

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car=car, img_size=img_size, split=which_split, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, augment=augment)

####################################################################################################################################################

class SVIROUncertainty(BaseDatasetCar):
    """
    https://sviro.kl.dfki.de
    
    Make sure to have a folder structure as follows:

    SVIRO-Illumination 
    └── sharan
        ├── train
        ├── test-adults
        ├── test-objects
        └── test-adults-and-objects

    """
    def __init__(self, car, img_size, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "SVIRO-Uncertainty"

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car=car, img_size=img_size, split=which_split, make_scene_impossible=False, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)

    def _get_data(self):

        # get all the png files, i.e. experiments
        self.files = sorted(list(self.root_dir.glob(f"{self.car}/pp_{self.split}_{self.img_size}/*/ir.png")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []

        # for each file
        for file in self.files:
        
            # get classification labels for each seat from folder name
            classif_labels = self._get_classif_label(file.parent)

            # open the image specified by the path
            # make sure it is a grayscale image
            img = np.array(Image.open(file).convert("L"))

            # each scene will be an array of images
            # append the image to the placeholder
            self.images.append([img])
        
            # append label to placeholder
            self.labels.append(classif_labels)

    def _pre_process_dataset(self):

        # get all the subfolders inside the dataset folder
        data_folder_variations = self.root_dir.glob("*")

        # for each variation
        for folder in data_folder_variations:

            # for each split
            for pre_processed_split in [f"pp_train-adults_{self.img_size}", f"pp_train-adults-and-seats_{self.img_size}", f"pp_test-adults_{self.img_size}", f"pp_test-objects_{self.img_size}", f"pp_test-seats_{self.img_size}", f"pp_test-adults-and-objects_{self.img_size}", f"pp_test-adults-and-seats_{self.img_size}", f"pp_test-adults-and-seats-and-objects_{self.img_size}"]:

                # create the path
                path_to_preprocessed_split = folder / pre_processed_split
                path_to_vanilla_split = folder / pre_processed_split.split("_")[1]

                # if no pre-processing for these settings exists, then create them
                if not path_to_preprocessed_split.exists():

                    print("-" * 37)
                    print(f"Pre-process and save data for folder: {folder} and split: {pre_processed_split} and downscale size: {self.img_size} ...")
                    
                    self.pre_process_and_save_data(path_to_preprocessed_split, path_to_vanilla_split)
                    
                    print("Pre-processing and saving finished.")
                    print("-" * 37)

    def pre_process_and_save_data(self, path_to_preprocessed_split, path_to_vanilla_split):
        """
        To speed up training, it is beneficial to do the rudementary pre-processing once and save the data.
        """

        # create the folders to save the pre-processed data
        path_to_preprocessed_split.mkdir()

        # get all the files in all the subfolders 
        files = list(path_to_vanilla_split.glob(f"**/ir.png")) + list(path_to_vanilla_split.glob(f"**/rgb.png"))

        # for each image 
        for curr_file in files:

            # open the image specified by the path
            img = Image.open(curr_file).convert("L") if "ir" in curr_file.name else Image.open(curr_file).convert("RGB")

            # center crop the image using the smaller size (i.e. width or height)
            # to define the new size of the image (basically we remove only either the width or height)
            img = TF.center_crop(img, np.min(img.size))

            # then resize the image to the one we want to use for training
            img = TF.resize(img, self.img_size)

            # create the folder for the experiment
            save_folder = path_to_preprocessed_split / curr_file.parent.stem
            save_folder.mkdir(exist_ok=True)

            # save the processed image
            img.save(save_folder / curr_file.name)

    def _get_positive(self, positive_label):

        # get all the potential candidates from the real images which have the same label as the synthetic one
        masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        input_image = self.images[index][0]
        input_image = TF.to_tensor(input_image)

        return input_image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index][0]
        label = self.labels[index]
        str_label = self._get_classif_str(label)

        # transform it for pytorch (normalized and transposed)
        image = TF.to_tensor(image)
    
        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt": self.label_str_to_int[str_label]}


####################################################################################################################################################


class ORSS(BaseDatasetCar):

    def __init__(self, car, img_size, which_split, make_instance_impossible, augment):

        # path to the main folder
        root_dir = Path(f"/data/workingdir/sds/data/ORSS/")

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car=car, img_size=img_size, split=which_split, make_scene_impossible=False, make_instance_impossible=make_instance_impossible, augment=augment)
        
    def _get_data(self):

        # get all folders with the sceneries
        if self.split == "all":
            if self.car.lower() == "all":
                self.files = sorted(list(self.root_dir.glob(f"*/*_pp_640_{self.img_size}/*")))
            else:
                self.files = sorted(list(self.root_dir.glob(f"{self.car}/*_pp_640_{self.img_size}/*")))
        else:
            if self.car.lower() == "all":
                self.files = sorted(list(self.root_dir.glob(f"*/{self.split}_pp_640_{self.img_size}/*")))
            else:
                self.files = sorted(list(self.root_dir.glob(f"{self.car}/{self.split}_pp_640_{self.img_size}/*")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []
        self.ids = []

        # for each file
        for file in self.files:

            # remove faulty label
            if "6_1_0" in file.name:
                continue

            # get classification labels for each seat from folder name
            classif_labels, image_id = self._get_classif_label(file)
        
            # open the image specified by the path
            # make sure it is a grayscale image
            img = np.array(Image.open(file).convert("L"))

            # append the image to the placeholder
            self.images.append([img])

            # append label and id to placeholder
            self.labels.append(classif_labels)
            self.ids.append(image_id)

    def _pre_process_dataset(self):
        pass

    def pre_process_and_save_data(self, *args, **kwargs):
        pass

    def _get_classif_label(self, file_path):

        # get the filename only of the path
        name = file_path.stem

       # split the elements at _
       # get the last three
        gts = name.split("_")[-3:]
        
        # convert the strings to ints
        gts = [int(x) for x in gts]

        # get the image id
        image_id = int(name.split("_")[1])

        return gts, image_id

    def _get_positive(self, positive_label, id=None):

        if id is None:
            masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 
        else:
            # get all the potential candidates from the real images which have the same label as the synthetic one
            masked = [idx for idx, (x,y) in enumerate(zip(self.labels, self.ids)) if (x==positive_label and (y<id-50 or y>id+50))] 

            if not masked:
                masked = [idx for idx, (x,y) in enumerate(zip(self.labels, self.ids)) if (x==positive_label and (y<id-25 or y>id+25))] 
                if not masked:
                    masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        input_image = self.images[index][0]
        input_image = TF.to_tensor(input_image)

        return input_image, self.ids[index]

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index][0]
        label = self.labels[index]
        id = self.ids[index]

        # transform it for pytorch (normalized and transposed)
        image = TF.to_tensor(image)

        if self.make_instance_impossible:
            output_image, _ = self._get_positive(label, id)
        else:
            output_image = image.clone()

        # randomly flip
        if random.choice([True, False]):
            image = TF.hflip(image)
            output_image = TF.hflip(output_image)
            label = label[::-1]

        str_label = self._get_classif_str(label)

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt": self.label_str_to_int[str_label]}

####################################################################################################################################################

class Fashion(TFashionMNIST):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, img_size, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/workingdir/sds/data/")

        # train or test split
        self.img_size = img_size
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, train=self.is_train, download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def _get_subset_of_data(self):

        self.images = self.data
        self.labels = self.targets

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, label in enumerate(self.labels):
                
                # make sure its a string
                label =  self._get_classif_str(label)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
            
            # only take the subset of indices based on how many samples per class to keep
            self.data = [x for idx, x in enumerate(self.images) if idx in keep_indices]
            self.targets = [x for idx, x in enumerate(self.labels) if idx in keep_indices]

    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.targets)-1)
            if int(self.targets[index]) == positive_label:
                image = self.data[index]
                image = Image.fromarray(image.numpy(), mode='L')
                image = TF.resize(image, [self.img_size, self.img_size])
                image = TF.to_tensor(image)

                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.data[index]
        label = int(self.targets[index])

        # doing this so that it is consistent with all other datasets to return a PIL Image
        image = Image.fromarray(image.numpy(), mode='L')

        # transform it for pytorch (normalized and transposed)
        image = TF.resize(image, [self.img_size, self.img_size])
        image = TF.to_tensor(image)

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class MNIST(TMNIST):

    def __init__(self, img_size, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/workingdir/sds/data/MNIST")

        # train or test split, digits or letters
        self.img_size = img_size
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, train=self.is_train, download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # dict to transform integers to string labels 
        self.int_to_label_str = {x:str(x) for x in range(10)}

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, label in enumerate(self.targets):
                
                # make sure its a string
                label =  self._get_classif_str(label)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self.targets)]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
            # get the image
            current_image = Image.fromarray(self.data[idx].numpy(), mode="L")

            # transform it for pytorch (normalized and transposed)
            current_image = TF.resize(current_image, [self.img_size, self.img_size])
            current_image = TF.to_tensor(current_image)

            # get label
            current_label = self.targets[idx]

            # keep it
            self.images.append(current_image)
            self.labels.append(current_label)

        del self.targets
        del self.data
            
    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class FONTS(Dataset):

    string_labels_to_integer_dict = dict()

    def __init__(self, which_factor, img_size, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # digits or letters
        self.factor = which_factor.lower()

        # train or test
        self.img_size = img_size
        self.is_train = True if which_split.lower() == "train" else False

        # path to the main folder
        self.root_dir = Path(f"/data/workingdir/sds/data/") / "REAL_FONTS" / self.factor

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        # number of images to use
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = list(self.root_dir.glob("*/*.png"))

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, img in enumerate(self.images):
                
                # get the label 
                label = self._get_label_from_path(img)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
            
            # only take the subset of indices based on how many samples per class to keep
            self.images = [x for idx, x in enumerate(self.images) if idx in keep_indices]

    def _get_label_from_path(self, path):

        # make sure its a string
        label =  ord(str(path.parent.name))
        
        # if it is from a letter, the order should be 
        # above 96, otherwise starting at 48
        if label > 96:
            label = label - 97
        else:
            label = label - 48

        return label 

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = Image.open(self.images[index]).convert("L")
        label = self._get_label_from_path(self.images[index])

        # transform it for pytorch (normalized and transposed)
        image = TF.resize(image, [self.img_size, self.img_size])
        image = TF.to_tensor(image)

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class GTSRB(Dataset):

    string_labels_to_integer_dict = dict()

    def __init__(self, img_size, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # train or test
        self.img_size = img_size
        self.is_train = True if which_split.lower() == "train" else False

        if which_split.lower() == "train":
            self.folder = "train_png"
        elif which_split.lower() == "test":
            self.folder = "test_png"
        elif which_split.lower() == "ood":
            self.folder = "ood_png"
        else:
            raise ValueError

        # path to the main folder
        self.root_dir = Path(f"/data/workingdir/sds/data/GTSRB") / self.folder

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # only get a subset of the data
        self._get_subset_of_data()

        # dict to transform integers to string labels 
        self.int_to_label_str = {x:str(x) for x in range(10)}

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.RandomBrightnessContrast(always_apply=False, p=0.4, brightness_limit=(0.0, 0.33), contrast_limit=(0.0, 0.33), brightness_by_max=False),
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                    # album.CoarseDropout(always_apply=False, p=0.4, fill_value=0.5, min_holes=5, max_holes=5, min_height=12, max_height=12, min_width=12, max_width=12),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        # number of images to use
        return len(self.images)

    def _get_subset_of_data(self):

        self.all_images = list(self.root_dir.glob("*/*.png"))
        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, img in enumerate(self.all_images):
                
                # get the label 
                label = self._get_label_from_path(img)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self.all_images)]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
                # get the image
                current_image = Image.open(self.all_images[idx]).convert("L")

                # transform it for pytorch (normalized and transposed)
                current_image = TF.resize(current_image, [self.img_size, self.img_size])
                current_image = TF.to_tensor(current_image)

                # get label
                current_label = self._get_label_from_path(self.all_images[idx])

                # keep it
                self.images.append(current_image)
                self.labels.append(current_label)

    def _get_label_from_path(self, path):

        # get the name from the parent folder
        if self.folder == "ood":
            if int(path.parent.name) < 10:
                return int(path.parent.name)
            else:
                return int(path.parent.name)-10
        else:
            return int(path.parent.name)-10

    def _get_positive(self, positive_label):

        # get all the potential candidates from the real images which have the same label as the synthetic one
        masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        input_image = self.images[index]

        return input_image
        
    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = self.labels[index]

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class CIFAR10(TCIFAR10):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, img_size, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/workingdir/sds/data/CIFAR10")

        # train or test split
        self.img_size = img_size
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, train=self.is_train, download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.RandomBrightnessContrast(always_apply=False, p=0.4, brightness_limit=(0.0, 0.33), contrast_limit=(0.0, 0.33), brightness_by_max=False),
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, label in enumerate(self.targets):
                
                # make sure its a string
                label =  self._get_classif_str(label)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self.data)]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
            # get the image
            current_image = Image.fromarray(self.data[idx]).convert("L")

            # transform it for pytorch (normalized and transposed)
            current_image = TF.resize(current_image, [self.img_size, self.img_size])
            current_image = TF.to_tensor(current_image)

            # get label
            current_label = self.targets[idx]

            # keep it
            self.images.append(current_image)
            self.labels.append(current_label)

        del self.targets
        del self.data
            
    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################


class SVHN(TSVHN):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, img_size, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/workingdir/sds/data/SVHN")

        # train or test split
        self.img_size = img_size
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, split="train" if self.is_train else "test", download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.RandomBrightnessContrast(always_apply=False, p=0.4, brightness_limit=(0.0, 0.33), contrast_limit=(0.0, 0.33), brightness_by_max=False),
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.targets = self.labels
        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, label in enumerate(self.targets):
                
                # make sure its a string
                label =  self._get_classif_str(label)

                # increase the counter for this label
                counter[label] += 1
                
                # if we are above the theshold for this label
                if counter[label] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self.data)]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
            # get the image
            current_image = Image.fromarray(np.transpose(self.data[idx], (1, 2, 0))).convert("L")

            # transform it for pytorch (normalized and transposed)
            current_image = TF.resize(current_image, [self.img_size, self.img_size])
            current_image = TF.to_tensor(current_image)

            # get label
            current_label = self.targets[idx]

            # keep it
            self.images.append(current_image)
            self.labels.append(current_label)

        del self.targets
        del self.data
            
    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class Omniglot(TOmniglot):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, img_size, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/workingdir/sds/data/Omniglot")

        # train or test split
        self.img_size = img_size
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, background=self.is_train, download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, (_, character_class) in enumerate(self._flat_character_images):
                
                # increase the counter for this label
                counter[character_class] += 1
                
                # if we are above the theshold for this label
                if counter[character_class] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self._flat_character_images)]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
            # get the image
            image_name, character_class = self._flat_character_images[idx]
            image_path = os.path.join(self.target_folder, self._characters[character_class], image_name)
            current_image = Image.open(image_path, mode='r').convert('L')

            # transform it for pytorch (normalized and transposed)
            current_image = TF.resize(current_image, [self.img_size, self.img_size])
            current_image = TF.to_tensor(current_image)

            # keep it
            self.images.append(current_image)
            self.labels.append(character_class)

    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class Places365(TPlaces365):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, img_size, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/workingdir/sds/data/Places365")

        # train or test split
        self.img_size = img_size
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, split="train-standard" if self.is_train else "val", small=True, download=False)

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx, (_, target) in enumerate(self.imgs):
                
                # increase the counter for this label
                counter[target] += 1
                
                # if we are above the theshold for this label
                if counter[target] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx, _ in enumerate(self.imgs)]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:
                
            # get the image
            file, target = self.imgs[idx]
            current_image = self.loader(file)

            # transform it for pytorch (normalized and transposed)
            current_image = TF.rgb_to_grayscale(current_image, num_output_channels=1)
            current_image = TF.resize(current_image, [self.img_size, self.img_size])
            current_image = TF.to_tensor(current_image)

            # keep it
            self.images.append(current_image)
            self.labels.append(target)

    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class LSUN(TLSUN):

    # dict to transform integers to string labels 
    int_to_label_str = {x:str(x) for x in range(10)}

    def __init__(self, img_size, which_split, make_instance_impossible, nbr_of_samples_per_class, augment):

        # path to the main folder
        root_dir = Path(f"/data/workingdir/sds/data/LSUN")

        # train or test split
        self.img_size = img_size
        self.split = which_split
        self.is_train = True if self.split.lower() == "train" else False

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # number of samples to keep for each class
        self.nbr_of_samples_per_class = nbr_of_samples_per_class

        # call the init function of the parent class
        super().__init__(root=root_dir, classes="train" if self.is_train else "test")

        # only get a subset of the data
        self._get_subset_of_data()

        # augmentations if needed
        if augment:
            self.augment = album.Compose(
                [
                    album.Blur(always_apply=False, p=0.4, blur_limit=7),
                    album.MultiplicativeNoise(always_apply=False, p=0.4, elementwise=True, multiplier=(0.75, 1.25)),
                ]
            )
        else:
            self.augment = False

    def _get_classif_str(self, label):
        return int(label)

    def __len__(self):
        return len(self.images)

    def _get_subset_of_data(self):

        self.images = []
        self.labels = []

        # if we are training
        if self.nbr_of_samples_per_class > 0:

            # keep track of samples per class
            counter = defaultdict(int)
            # and the corresponding indices
            keep_indices = []

            # for each label
            for idx in range(self.length):

                target = 0
                for ind in self.indices:
                    if idx < ind:
                        break
                    target += 1
                
                # increase the counter for this label
                counter[target] += 1
                
                # if we are above the theshold for this label
                if counter[target] >= (self.nbr_of_samples_per_class+1):
                    # then skip it
                    continue
                else:
                    # otherwise keep track of the label
                    keep_indices.append(idx)
        # testing
        else:
            keep_indices = [idx for idx in range(10_000)]

        # only take the subset of indices based on how many samples per class to keep
        for idx in keep_indices:

            target = 0
            sub = 0
            for ind in self.indices:
                if idx < ind:
                    break
                target += 1
                sub = ind
                
            db = self.dbs[target]
            idx = idx - sub

            current_image, _ = db[idx]

            # transform it for pytorch (normalized and transposed)
            current_image = TF.rgb_to_grayscale(current_image, num_output_channels=1)
            current_image = TF.resize(current_image, [self.img_size, self.img_size])
            current_image = TF.to_tensor(current_image)

            # keep it
            self.images.append(current_image)
            self.labels.append(target)

    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.labels)-1)
            if int(self.labels[index]) == positive_label:
                image = self.images[index]
                return image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index]
        label = int(self.labels[index])

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        # augment image if necessary (we need 0-channel input, not 1-channel input)
        if self.augment: image = torch.from_numpy(self.augment(image=np.array(image)[0])["image"][None,:])

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class SynthAndRealCar(BaseDatasetCar): 

    def __init__(self, car, img_size, which_split, make_scene_impossible, make_instance_impossible, augment):

        # get the real and synthetic dataset
        self.real_dataset = ORSS(car="X5", img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, augment=augment)
        if car == "uncertainty":
            self.synthetic_dataset = SVIROUncertainty(car="sharan", img_size=img_size, which_split="*", make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=-1, augment=augment)
        else:
            self.synthetic_dataset = Seats_And_People_Nocar(car="cayenne", img_size=img_size, which_split=which_split, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, augment=augment)

        # placeholder for all labels and images, but masked
        self.synth_labels = []
        self.synth_images = []
        self.real_labels = []
        self.real_images = []
        self.real_ids = []

        # get only the real images which have labels in the synthetic dataset
        for real_labels, real_images, real_id in zip(self.real_dataset.labels, self.real_dataset.images, self.real_dataset.ids):
            if real_labels in self.synthetic_dataset.labels:
                if real_labels[0] == 0 and real_labels[1] == 0 and real_labels[2] == 0:
                    continue
                self.real_labels.append(real_labels)
                self.real_images.append(real_images)
                self.real_ids.append(real_id)

        # now get only the synthetic images for which there are also real images
        for synth_labels, synth_images in zip(self.synthetic_dataset.labels, self.synthetic_dataset.images):
            if synth_labels in self.real_labels:
                if synth_labels[0] == 0 and synth_labels[1] == 0 and synth_labels[2] == 0:
                    continue
                self.synth_labels.append(synth_labels)
                self.synth_images.append(synth_images)

        # get it from sythetic dataset
        self.make_scene_impossible =  self.synthetic_dataset.make_scene_impossible
        self.make_instance_impossible =  self.synthetic_dataset.make_instance_impossible
        self.label_str_to_int = self.synthetic_dataset.label_str_to_int
        self.int_to_label_str = self.synthetic_dataset.int_to_label_str
        self.augment =  self.synthetic_dataset.augment

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.synth_images)

    def _get_real_positive(self, label, id=None):

        if id is None:
            masked = [idx for idx, x in enumerate(self.real_labels) if x==label] 
        else:
            # get all the potential candidates from the real images which have the same label as the synthetic one
            masked = [idx for idx, (x,y) in enumerate(zip(self.real_labels, self.real_ids)) if (x==label and (y<id-50 or y>id+50))] 

            if not masked:
                masked = [idx for idx, (x,y) in enumerate(zip(self.real_labels, self.real_ids)) if (x==label and (y<id-25 or y>id+25))] 
                if not masked:
                    masked = [idx for idx, x in enumerate(self.real_labels) if x==label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        input_image = self.real_images[index][0]

        return input_image

    def _get_real(self, label):

        # get all the potential candidates from the real images which have the same label as the synthetic one
        masked = [idx for idx, x in enumerate(self.real_labels) if x==label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        # we take some random samples from the selected scene to avoid using the same images
        # for simplicity we take the same random indices 
        input_image = self.real_images[index][0]
        id = self.real_ids[index]

        if self.make_instance_impossible:
            output_image = self._get_real_positive(label, id)
        else:
            output_image = self.real_images[index][0]

        return input_image, output_image

    def _get_synth_positive(self, positive_label):

        # get all the potential candidates from the real images which have the same label as the synthetic one
        masked = [idx for idx, x in enumerate(self.synth_labels) if x==positive_label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        input_image = self.synth_images[index][0]

        return input_image

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        synthetic_images = self.synth_images[index]
        synthetic_label = self.synth_labels[index]

        # get the images
        real_input_image, real_output_image = self._get_real(synthetic_label)

        real_input_image =  TF.to_tensor(real_input_image)
        real_output_image =  TF.to_tensor(real_output_image)

        # randomly selected
        # .) the input images 
        # .) the output images 
        if len(synthetic_images) > 1:
            rand_indices = random.sample(range(0,len(synthetic_images)), 2)
        else:
            rand_indices = [0,0]

        # get the image to be used as input
        synthetic_input_image = synthetic_images[rand_indices[0]]
        synthetic_input_image =  TF.to_tensor(synthetic_input_image)

        # randomly flip
        to_flip = random.choice([True, False])
        if to_flip:
            if synthetic_label[::-1] in self.real_labels and synthetic_label[::-1] in self.synth_labels:
                synthetic_input_image = TF.hflip(synthetic_input_image)
                real_input_image = TF.hflip(real_input_image)
                real_output_image = TF.hflip(real_output_image)
                synthetic_label = synthetic_label[::-1]
            else:
                to_flip = False

        str_synthetic_label = self._get_classif_str(synthetic_label)

        # get the image to be used for the reconstruction error
        if self.make_scene_impossible:
            synthetic_output_image = synthetic_images[rand_indices[1]]
            synthetic_output_image =  TF.to_tensor(synthetic_output_image)
            if to_flip:
                synthetic_output_image = TF.hflip(synthetic_output_image)
        elif self.make_instance_impossible:
            synthetic_output_image = self._get_synth_positive(synthetic_label)
            synthetic_output_image =  TF.to_tensor(synthetic_output_image)
        else:
            synthetic_output_image = synthetic_images[rand_indices[0]]
            synthetic_output_image =  TF.to_tensor(synthetic_output_image)
            if to_flip:
                synthetic_output_image = TF.hflip(synthetic_output_image)

        return {
            "image":synthetic_input_image, 
            "target":synthetic_output_image, 
            "real_image":real_input_image, 
            "real_target":real_output_image, 
            "gt": self.label_str_to_int[str_synthetic_label], 
            }

####################################################################################################################################################


class MPI3D(Dataset):
    """
    https://github.com/rr-learning/disentanglement_dataset

    Datasets: toy, realistic, real

    # shape             (1036800, 64, 64, 3)
    # reshape           (6,6,2,3,3,40,40,64,64,3)

    # object_color  	white=0, green=1, red=2, blue=3, brown=4, olive=5
    # object_shape 	    cone=0, cube=1, cylinder=2, hexagonal=3, pyramid=4, sphere=5
    # object_size 	    small=0, large=1
    # camera_height 	top=0, center=1, bottom=2
    # background_color 	purple=0, sea green=1, salmon=2
    # horizontal_axis 	0,...,39
    # vertical_axis 	0,...,39

    Make sure to have a folder structure as follows:

    MPI3D 
    ├── realistic.npz
    ├── toy.npz
    └── real.npz
    """
    def __init__(self, which_dataset, img_size, which_factor, augment):

        assert img_size == 64

        # path to the main folder
        self.root_dir = ROOT_DATA_DIR / "MPI3D"

        # which dataset is being used
        self.which_dataset = which_dataset

        # select which dataset to use
        self.data_dir = self.root_dir / f"{self.which_dataset}.npz"
     
        # which factors to use from the dataset
        self.which_factor = which_factor

        # load the data as images to the memory
        self._get_data()
        self.string_labels_to_int()

        self.augment = augment

        # we only need these transformation for training
        if self.augment:
            if self.augment:
                self.augment = album.Compose([
                    album.GaussNoise(),    
                    album.ISONoise(),  
                ], p=1.0)

    def _get_data(self):

        # load the numpy array from the file
        self.images = np.load(self.data_dir)['images']
        self.images = self.images.reshape(6,6,2,3,3,40,40,64,64,3)
        self.reshaped_images_size = self.images.shape

        # load the factors we need
        if self.which_factor == "reduced":
            self.images = self.images[:,:,1,:,:,:,:,:,:,:] # get only large and camera center images, background_color sea green
            self.reshaped_images_size = self.images.shape
            
        self.images = self.images.reshape(-1, 64, 64, 3)

    def string_labels_to_int(self):

        # empty dict
        self.string_labels_to_integer_dict = dict()

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image
        input_image = self.images[index]
        output_image =  TF.to_tensor(input_image)

        # augment input image if needed
        if self.augment:
            input_image = self.augment(image=np.array(input_image))['image']
        
        # make sure its a tensor
        input_image =  TF.to_tensor(input_image)

        return {"image":input_image, "target":output_image}

####################################################################################################################################################

class TICAM(BaseDatasetCar):
    """
    https://vizta-tof.kl.dfki.de/

    
    You only need the IR images.
    Make sure to have a folder structure as follows:

    TICaM 
    ├── train
    │   ├── cs00
    │   ├── cs01
    │   ├── cs02
    │   ├── cs03
    │   └── cs04
    └── test
        ├── cs01
        ├── cs02
        ├── cs03
        ├── cs04
        └── cs05
    """
    def __init__(self, img_size):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "TICaM" 

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car="all", split="all", img_size=img_size, make_scene_impossible=False, make_instance_impossible=False)

    def _get_data(self):

        # get all png images
        self.files = sorted(list(self.root_dir.glob(f"pp_*_{self.img_size}/**/*.png")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []

        # for each file
        for counter, file in enumerate(self.files):

            # get classification labels for each seat from file name
            classif_labels = self._get_classif_label(file)

            # each scene will be an array of images
            self.images.append([])
        
            # open the image specified by the path
            img = Image.open(file).convert("L")

            # append the image to the placeholder
            self.images[counter].append(img)

             # append label to placeholder
            self.labels.append(classif_labels)

    def _get_classif_label(self, file_path):

        # get the filename only of the path
        name = file_path.stem

        # split the elements at _
        file_name_parts = name.split("_")

        # get the two entries responsible for the classification of the scenery
        # add 0 label in the middle to be consistent with the other datasets
        # left, middle (not existing), right
        gts = [file_name_parts[-6], 0, file_name_parts[3]]

        # convert the string definitions into integers
        for idx, value in enumerate(gts):
            if value == 0:
                continue
            # adult passenger
            elif "p" in value:
                gts[idx] = 3
            # everyday objects
            elif "o" in value:
                gts[idx] = 0
            # infant seat
            elif value in ["s05", "s15", "s06", "s16"]:
                gts[idx] = 1
            # child seat
            elif value in ["s03", "s13", "s04", "s14"]:
                gts[idx] = 2
            elif value in ["s01", "s11", "s02", "s12"]:
                if file_name_parts[-3] == "g00":
                    gts[idx] = 2
                elif file_name_parts[-3] == "g01" or file_name_parts[-3] == "g11" or file_name_parts[-3] == "g10":
                    gts[idx] = 1
        
        # convert the strings to ints
        gts = [int(x) for x in gts]

        return gts

    def _pre_process_dataset(self):

        # for each split
        for pre_processed_split in [f"pp_train_{self.img_size}", f"pp_test_{self.img_size}"]:

            # create the path
            path_to_preprocessed_split = self.root_dir / pre_processed_split
            path_to_vanilla_split = self.root_dir / pre_processed_split.split("_")[1]

            # if no pre-processing for these settings exists, then create them
            if not path_to_preprocessed_split.exists():

                print("-" * 37)
                print(f"Pre-process and save data for folder: {self.root_dir} and split: {pre_processed_split} and downscale size: {self.img_size} ...")
                
                self.pre_process_and_save_data(path_to_preprocessed_split, path_to_vanilla_split)
                
                print("Pre-processing and saving finished.")
                print("-" * 37)

    def pre_process_and_save_data(self, path_to_preprocessed_split, path_to_vanilla_split):
        """
        To speed up training, it is beneficial to do the rudementary pre-processing once and save the data.
        """

        # create the folders to save the pre-processed data
        path_to_preprocessed_split.mkdir()

        # get all the files in all the subfolders 
        files = list(path_to_vanilla_split.glob(f"**/*.png"))

        # for each image 
        for curr_file in files:

            # open the image specified by the path
            img = cv2.imread(str(curr_file), -1) 

            # histogram equalization
            img = exposure.equalize_hist(img)

            # make it a tensor
            img = torch.from_numpy(img)

            # center crop the image using the smaller size (i.e. width or height)
            # to define the new size of the image (basically we remove only either the width or height)
            # img = TF.center_crop(img, 300)
            img = TF.crop(img, top=120, left=(512-300)//2, height=300, width=300)

            # then resize the image to the one we want to use for training
            img = TF.resize(img.unsqueeze(0), self.img_size)

            # make it a pil again
            img = TF.to_pil_image(img)

            # create the folder for the experiment
            save_folder = path_to_preprocessed_split / curr_file.parent.stem
            save_folder.mkdir(exist_ok=True)

            # save the processed image
            img.save(save_folder / curr_file.name)
        
    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index][0]
        label = self.labels[index]

        # transform it for pytorch (normalized and transposed)
        image =  TF.to_tensor(image)

        return {"image":image, "target":image, "gt_left": label[0], "gt_middle": label[1], "gt_right": label[2]}

####################################################################################################################################################

def print_dataset_statistics(dataset, which_dataset, which_split):

    # if a vehicle dataset
    if which_dataset.lower() in ["sviro", "sviro_illumination", "sviro_uncertainty", "seats_and_people_nocar", "orss"]:

        # get the int label for all labels
        labels = np.array([dataset.label_str_to_int["_".join([str(y) for y in x])] for x in dataset.labels])
        int_to_label_str = dataset.int_to_label_str

    elif which_dataset.lower() in ["synth_and_real_car", "synth_and_real_car_uncertainty"]:

        labels = np.array([dataset.label_str_to_int["_".join([str(y) for y in x])] for x in dataset.synth_labels])
        int_to_label_str = dataset.int_to_label_str
    
    elif hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
        int_to_label_str = None

    elif hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
        int_to_label_str = None
    
    else:
        print("No targets or labels attribute.")
        return 

    unique_labels, labels_counts = np.unique(labels, return_counts=True)

    if int_to_label_str is None:
        int_to_label_str = {x:str(x) for x in unique_labels}

    print("=" * 37)
    print("Dataset used: \t", dataset)
    print("Split: \t\t", which_split)
    print("Samples: \t", len(dataset))
    print("-" * 37)

    # print the label and its number of occurences
    for label, count in zip(unique_labels, labels_counts):
        print(f"Label {int_to_label_str[label]}: {count}")

    print("=" * 37)

####################################################################################################################################################

def create_dataset(which_dataset, which_factor, img_size, which_split, make_scene_impossible=False, make_instance_impossible=False, augment=False, batch_size=64, shuffle=True, nbr_of_samples_per_class=-1, print_dataset=True):

    # create the dataset
    if which_dataset.lower() == "sviro":
        dataset = SVIRO(car=which_factor, img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, augment=augment)
    elif which_dataset.lower() == "sviro_illumination":
        dataset = SVIROIllumination(car=which_factor, img_size=img_size, which_split=which_split, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, augment=augment)
    elif which_dataset.lower() == "sviro_uncertainty":
        dataset = SVIROUncertainty(car=which_factor, img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "seats_and_people_nocar":
        dataset = Seats_And_People_Nocar(car="cayenne", img_size=img_size, which_split=which_split, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, augment=augment)
    elif which_dataset.lower() == "orss":
        dataset = ORSS(car=which_factor, img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, augment=augment)
    elif which_dataset.lower() == "fashion":
        dataset = Fashion(img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "mnist":
        dataset = MNIST(img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "fonts":
        dataset = FONTS(which_factor=which_factor, img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "gtsrb":
        dataset = GTSRB(img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "cifar10":
        dataset = CIFAR10(img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "svhn":
        dataset = SVHN(img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "omniglot":
        dataset = Omniglot(img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "places365":
        dataset = Places365(img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "lsun":
        dataset = LSUN(img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
    elif which_dataset.lower() == "synth_and_real_car":
        dataset = SynthAndRealCar(car=which_factor, img_size=img_size, which_split=which_split, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, augment=augment)
    elif which_dataset.lower() == "synth_and_real_car_uncertainty":
        dataset = SynthAndRealCar(car="uncertainty", img_size=img_size, which_split=which_split, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, augment=augment)
    elif which_dataset.lower() == "orss_and_synth_nocar":
        datasets = [
            ORSS(car="X5", img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, augment=augment),
            Seats_And_People_Nocar(car="cayenne", img_size=img_size, which_split=which_split, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, augment=augment)
        ]
        dataset = ConcatDataset(datasets)
        dataset.label_str_to_int = dataset.datasets[1].label_str_to_int
        dataset.int_to_label_str = dataset.datasets[1].int_to_label_str
    elif which_dataset.lower() == "orss_and_synth_uncertainty":
        datasets = [
            ORSS(car="X5", img_size=img_size, which_split=which_split, make_instance_impossible=make_instance_impossible, augment=augment),
            SVIROUncertainty(car="sharan", img_size=img_size, which_split="*", make_instance_impossible=make_instance_impossible, nbr_of_samples_per_class=nbr_of_samples_per_class, augment=augment)
        ]
        dataset = ConcatDataset(datasets)
        dataset.label_str_to_int = dataset.datasets[1].label_str_to_int
        dataset.int_to_label_str = dataset.datasets[1].int_to_label_str
    elif which_dataset.lower() == "mpi3d":
        dataset = MPI3D(which_dataset=which_factor, img_size=img_size, which_factor="reduced", augment=augment)
    elif which_dataset.lower() == "ticam":
        dataset = TICAM(img_size=img_size)
    else:
        raise ValueError

    if len(dataset) == 0:
        raise ValueError("The length of the dataset is zero. There is probably a problem with the folder structure for the dataset you want to consider. Have you downloaded the dataset and used the correct folder name and folder tree structure?")

    # for reproducibility
    # https://pytorch.org/docs/1.9.0/notes/randomness.html?highlight=reproducibility
    g = torch.Generator()
    g.manual_seed(0)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # create loader for the defined dataset
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    if print_dataset:
        print_dataset_statistics(dataset, which_dataset, which_split)

    return train_loader

####################################################################################################################################################