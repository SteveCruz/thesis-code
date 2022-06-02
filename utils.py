##############################################################################################################################################################
##############################################################################################################################################################
"""
Helperfunctions used by other scripts.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import math
import toml
import time
import torch
import pickle
import datetime
import dateutil

import numpy as np

from pathlib import Path
from collections import defaultdict

from PIL import Image
from sklearn import metrics
from torchvision.utils import save_image as save_grid
from torchvision.transforms import functional as TF

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

##############################################################################################################################################################
##############################################################################################################################################################

class Tee(object):
    """
    Class to make it possible to print text to the console and also write the 
    output to a file.
    """

    def __init__(self, original_stdout, file):

        # keep the original stdout
        self.original_stdout = original_stdout

        # the file to write to
        self.log_file_handler= open(file, 'w')

        # all the files the print statement should be saved to
        self.files = [self.original_stdout, self.log_file_handler]

    def write(self, obj):

        # for each file
        for f in self.files:

            # write to the file
            f.write(obj)

            # If you want the output to be visible immediately
            f.flush() 

    def flush(self):

        # for each file
        for f in self.files:

            # If you want the output to be visible immediately
            f.flush()

    def end(self):

        # close the file
        self.log_file_handler.close()

        # return the original stdout
        return self.original_stdout

##############################################################################################################################################################

def write_toml_to_file(cfg, file_path):
    """
    Write the parser to a file.
    """

    with open(file_path, 'w') as output_file:
        toml.dump(cfg, output_file)

    print('=' * 57)
    print("Config file saved: ", file_path)
    print('=' * 57)
    
##############################################################################################################################################################

def save_accuracy(save_path, score):

    with save_path.open('w') as file:
        file.write(str(score))

def save_dict(save_path, score_dict):

    with save_path.open('w') as file:
        for key, value in score_dict.items():
            file.write(f"{key}:{value}\n") 

##############################################################################################################################################################

class TrainingTimer():
    """
    Keep track of training times.
    """

    def __init__(self):

        # get the current time
        self.start = datetime.datetime.fromtimestamp(time.time())
        self.eval_start = datetime.datetime.fromtimestamp(time.time())

        # print it human readable
        print("Training start: ", self.start)
        print('=' * 57)

    def print_end_time(self):

        # get datetimes for simplicity
        datetime_now = datetime.datetime.fromtimestamp(time.time())

        print("Training finish: ", datetime_now)

        # compute the time different between now and the input
        rd = dateutil.relativedelta.relativedelta(datetime_now, self.start)

        # print the duration in human readable
        print(f"Training duration: {rd.hours} hours, {rd.minutes} minutes, {rd.seconds} seconds")

    
    def print_time_delta(self):

        # get datetimes for simplicity
        datetime_now = datetime.datetime.fromtimestamp(time.time())

        # compute the time different between now and the input
        rd = dateutil.relativedelta.relativedelta(datetime_now, self.eval_start)

        # print the duration in human readable
        print(f"Duration since last evaluation: {rd.hours} hours, {rd.minutes} minutes, {rd.seconds} seconds")
        print('=' * 57)

        # update starting time
        self.eval_start = datetime.datetime.fromtimestamp(time.time())

##############################################################################################################################################################

def plot_progress(images, save_folder, nbr_epoch, text, max_nbr_samples=16, transpose=False, **kwargs):

    # check which number is smaller and use it to extract the correct amount of images per element in the list
    nbr_samples = min(images[0].shape[0], max_nbr_samples)

    # collect only the number of images we want to plot
    gather_sample_results = [x[:nbr_samples].detach() for x in images]
    
    # if we want to transpose the whole grid, e.g. plot from left to right instead of 
    # plotting from top to down
    if transpose:

        # collect the transposed version
        transposed_gather = []

        # for each sample, i.e. batch element
        for idx in range(nbr_samples):

            # collect from each list the idx'th sample
            # e.g. each reconstuction of batch element idx
            samples = [x[idx] for x in gather_sample_results]

            # make sure its a tensor and add the empty dimension for single channel image
            transposed_gather.append(torch.cat(samples).unsqueeze(1))
        
        # since we plot differently, we need to change the number of samples
        # to start the new row at the correct position
        nbr_samples = len(gather_sample_results)

        # change pointer for ease of use 
        gather_sample_results = transposed_gather

    # concat all the tensors in the list to a single tensor of the correct dimension
    gather_sample_results = torch.cat(gather_sample_results, dim=0)

    # create the filename
    save_name = save_folder["images"] / (text + "_" + str(nbr_epoch) + ".png")

    # save images as grid
    # each element in the batch has its own column
    save_grid(gather_sample_results, save_name, nrow=nbr_samples, **kwargs)

##############################################################################################################################################################

def plot_probability_results(results, labels, folders, nbr_recursions, nbr_trials):

    results_dict = defaultdict(dict) 

    for value, key in zip(results, labels):
        with open(folders["data-proba"] / f'{value}_tpr.pkl', 'rb') as f:
            results_dict[key]["tpr"] = pickle.load(f)
        with open(folders["data-proba"] / f'{value}_fpr.pkl', 'rb') as f:
            results_dict[key]["fpr"] = pickle.load(f)
        with open(folders["data-proba"] / f'{value}_precision.pkl', 'rb') as f:
            results_dict[key]["precision"] = pickle.load(f)
        with open(folders["data-proba"] / f'{value}_recall.pkl', 'rb') as f:
            results_dict[key]["recall"] = pickle.load(f)

    fig = plt.figure()
    for label, result in results_dict.items():
        plt.plot(result["fpr"], result["tpr"], linestyle='--', label=label)
    plt.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="-.", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.grid(axis="y", color="0.9", linestyle='-', linewidth=1)
    plt.box(False)
    plt.legend()
    fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
    fig.set_size_inches(PLT_WIDTH, PLT_HEIGHT)
    fig.savefig(folders["images-proba"] / (f"ROC-{nbr_recursions}_trials-{nbr_trials}.png"))
    fig.savefig(folders["images-proba"] / (f"ROC-{nbr_recursions}_trials-{nbr_trials}.pdf"))
    plt.close(fig=fig)

    fig = plt.figure()
    for label, result in results_dict.items():
        plt.plot(result["recall"], result["precision"], linestyle='--', label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(axis="y", color="0.9", linestyle='-', linewidth=1)
    plt.box(False)
    plt.legend()
    fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
    fig.set_size_inches(PLT_WIDTH, PLT_HEIGHT)
    fig.savefig(folders["images-proba"] / (f"PR-{nbr_recursions}_trials-{nbr_trials}.png"))
    fig.savefig(folders["images-proba"] / (f"PR-{nbr_recursions}_trials-{nbr_trials}.pdf"))
    plt.close(fig=fig)

##############################################################################################################################################################

def compute_scores(accurate_certain, accurate_uncertain, inaccurate_certain, inaccurate_uncertain, threshold_to_consider, which_dataset, folders, idx_recursion, which_classifier, fix_dropout_mask):

    # true_positive = accurate_certain = true certainty
    # true_negative = inaccurate_uncertain = true uncertainty 
    # false_positive = inaccurate_certain = false certainty
    # false_negative = accurate_uncertain = false uncertainty

    # tu / (tu + fc)
    # tc / (tc + fu)
    # tu / (tu + fu)
    # (tu + tc) / (tu + tc + fu + fc)

    # tpr = tp / (tp + fn)
    tpr = [accurate_certain[key]/(accurate_certain[key]+accurate_uncertain[key]) for key in sorted(threshold_to_consider)]
    # fpr = fp / (fp + tn)
    fpr = [inaccurate_certain[key]/(inaccurate_certain[key]+inaccurate_uncertain[key]) for key in sorted(threshold_to_consider)]
    
    with open(folders["data-all-proba"] / f'{which_dataset}_tpr_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(tpr, f)
    with open(folders["data-all-proba"] / f'{which_dataset}_fpr_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(fpr, f)

    # precision = tp / (tp + fp)
    precision = [accurate_certain[key]/(accurate_certain[key]+inaccurate_certain[key]) for key in sorted(threshold_to_consider)]
    # recall = tp / (tp + fn)
    recall = [accurate_certain[key]/(accurate_certain[key]+accurate_uncertain[key]) for key in sorted(threshold_to_consider)]

    # replace nan values with 1, otherwise aupr is always 0
    precision = [1 if math.isnan(x) else x for x in precision]

    with open(folders["data-all-proba"] / f'{which_dataset}_precision_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(precision, f)
    with open(folders["data-all-proba"] / f'{which_dataset}_recall_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(recall, f)

    save_accuracy(Path(folders["data-all-proba"]) / f"auroc_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", metrics.auc(fpr, tpr))
    save_accuracy(Path(folders["data-all-proba"]) / f"aupr_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", metrics.auc(recall, precision))

    tpr_rate = 0.3
    save_accuracy(Path(folders["data-all-proba"]) / f'fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.5
    save_accuracy(Path(folders["data-all-proba"]) / f'fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.8
    save_accuracy(Path(folders["data-all-proba"]) / f'fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.9
    save_accuracy(Path(folders["data-all-proba"]) / f'fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.95
    save_accuracy(Path(folders["data-all-proba"]) / f'fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    
##############################################################################################################################################################


def compute_ood_scores(true_positive, true_negative, false_positive, false_negative, threshold_to_consider, which_dataset, folders, idx_recursion, which_classifier, fix_dropout_mask):

    # tpr = tp / (tp + fn)
    tpr = [true_positive[key]/(true_positive[key]+false_negative[key]) for key in sorted(threshold_to_consider)]
    # fpr = fp / (fp + tn)
    fpr = [false_positive[key]/(false_positive[key]+true_negative[key]) for key in sorted(threshold_to_consider)]
    
    with open(folders["data-ood"] / f'{which_dataset}_tpr_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(tpr, f)
    with open(folders["data-ood"] / f'{which_dataset}_fpr_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(fpr, f)

    # precision = tp / (tp + fp)
    precision = [true_positive[key]/(true_positive[key]+false_positive[key]) for key in sorted(threshold_to_consider)]
    # recall = tp / (tp + fn)
    recall = [true_positive[key]/(true_positive[key]+false_negative[key]) for key in sorted(threshold_to_consider)]

    # replace nan values with 1, otherwise aupr is always 0
    precision = [1 if math.isnan(x) else x for x in precision]

    with open(folders["data-ood"] / f'{which_dataset}_precision_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(precision, f)
    with open(folders["data-ood"] / f'{which_dataset}_recall_recursion_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.pkl', 'wb') as f:
        pickle.dump(recall, f)

    save_accuracy(Path(folders["data-ood"]) / f"ood_auroc_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", metrics.auc(fpr, tpr))
    save_accuracy(Path(folders["data-ood"]) / f"ood_aupr_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance", metrics.auc(recall, precision))

    tpr_rate = 0.3
    save_accuracy(Path(folders["data-ood"]) / f'ood_fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.5
    save_accuracy(Path(folders["data-ood"]) / f'ood_fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.8
    save_accuracy(Path(folders["data-ood"]) / f'ood_fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.9
    save_accuracy(Path(folders["data-ood"]) / f'ood_fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))
    tpr_rate = 0.95
    save_accuracy(Path(folders["data-ood"]) / f'ood_fdpr-{tpr_rate}_{which_dataset}_{idx_recursion}_{which_classifier}_{fix_dropout_mask}.performance', np.nan_to_num(np.array(fpr).flat[np.abs(np.array(tpr) - tpr_rate).argmin()], nan=0.0))

##############################################################################################################################################################

def plot_img_to_xis(img, this_axis, title=None, title_color="black"):

    if not isinstance(img, np.ndarray):
        img = img.detach().cpu().numpy()

    if img.max() > 1:
        img = img / 255

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0) 

    img = np.moveaxis(img, 0, 2)

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    this_axis.imshow(img, vmin=0, vmax=1)
    this_axis.axis('off')

    if title is not None:
        this_axis.set_title(title, color=title_color)
