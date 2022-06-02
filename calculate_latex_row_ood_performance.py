####################################################################################################################################################
####################################################################################################################################################
"""
Compute performances and metrics (AUROC, AUPR, FPR) for several seed runs.
Make sure to provide 10 runs of the same kind of experiments.
Uncomment the hyperparameters you want to consider.
"""
####################################################################################################################################################
####################################################################################################################################################

import pickle
import toml
import numpy as np
from sklearn import metrics
from pathlib import Path
from collections import defaultdict

####################################################################################################################################################
####################################################################################################################################################

def get_row(experiment, which_classifier, fix_mask):

    assert len(experiments) == 10

    old_train = ""

    # keep track
    all_results = defaultdict(dict)

    for experiment in experiments:

        # get all the performance files in the data folder
        folder = Path("results") / experiment / "data" / "ood"
        files = list(folder.glob("*.pkl"))

        # get the config file to get the training data
        config_file = Path("results") / experiment / "cfg.toml"
        config = toml.load(config_file)
        train = config["dataset"]["name"]

        if old_train != "":
            if train != old_train:
                raise ValueError("Not all models are from the same dataset")


        # get only the files we are intersted in
        files = [x for x in files if (which_classifier in x.name or "None" in x.name) and (fix_mask in x.name or "None" in x.name)]

        # for each performance file
        for file in sorted(files):

            if "tpr" in str(file):
                metric = "tpr"
            elif "fpr" in str(file):
                metric = "fpr"
            elif "recall" in str(file):
                metric = "recall"
            elif "precision" in str(file):
                metric = "precision"
            else:
                continue

            # get the metric and the ood dataset from the filename
            dataset = file.name.split("_")[0].split("ood-")[-1]

            if dataset.lower()=="none" or dataset.lower()=="test" or dataset.lower()=="mlp":
                continue

            if not "ood" in dataset:
                dataset = "ood-"+dataset

            # get the value
            with file.open("rb") as f:
                values = pickle.load(f)

            # keep track of the value
            if metric in all_results[dataset]:
                all_results[dataset][metric].append(values)
            else:
                all_results[dataset][metric] = [values]

        # get all the performance files in the data folder
        folder = Path("results") / experiment / "data" / "all-proba"
        files = list(folder.glob("*.pkl"))
        files = [x for x in files if (which_classifier in x.name or "None" in x.name) and (fix_mask in x.name or "None" in x.name)]

        # for each performance file
        for file in sorted(files):

            if "tpr" in str(file):
                metric = "tpr"
            elif "fpr" in str(file):
                metric = "fpr"
            elif "recall" in str(file):
                metric = "recall"
            elif "precision" in str(file):
                metric = "precision"
            else:
                continue

            # get the metric and the ood dataset from the filename
            dataset = file.name.split("_")[0]

            if not "test" in dataset and not "tesla" in dataset:
                continue

            # get the value
            with file.open("rb") as f:
                values = pickle.load(f)

            # keep track of the value
            if metric in all_results[dataset]:
                all_results[dataset][metric].append(values)
            else:
                all_results[dataset][metric] = [values]
        
        old_train = train

    for dataset, values in all_results.items():
        for idx in range(len(experiments)):

            auroc = metrics.auc(values["fpr"][idx], values["tpr"][idx])
            aupr = metrics.auc(values["recall"][idx], values["precision"][idx])

            # tpr_rate = 0.5
            tpr_rate = 0.95
            fdpr = np.array(values["fpr"][idx]).flat[np.abs(np.array(values["tpr"][idx]) - tpr_rate).argmin()]

            auroc = np.nan_to_num(auroc, nan=0.0)
            aupr = np.nan_to_num(aupr, nan=0.0)
            fdpr = np.nan_to_num(fdpr, nan=0.0)

            auroc = 100 * auroc
            aupr = 100 * aupr
            fdpr = 100 * fdpr
            
            if "auroc" in values:
                values["auroc"].append(auroc)
            else:
                values["auroc"] = [auroc]
                
            if "aupr" in values:
                values["aupr"].append(aupr)
            else:
                values["aupr"] = [aupr]

            if "fdpr-0.95" in values:
                values["fdpr-0.95"].append(fdpr)
            else:
                values["fdpr-0.95"] = [fdpr]

    print("-"*77)
    print(experiment)
    print(f"Classifier: \t\t {which_classifier}")
    print(f"Dropout mask fixed: \t {fix_mask}")
    print(f"Training: \t\t {train}")
    print("-"*77)
    for dataset, values in all_results.items():

        values['auroc_mean'] = np.mean(np.array(values['auroc']))
        values['auroc_std'] = np.std(np.array(values['auroc']))
        values['aupr_mean'] = np.mean(np.array(values['aupr']))
        values['aupr_std'] = np.std(np.array(values['aupr']))
        values['fdpr-0.95_mean'] = np.mean(np.array(values['fdpr-0.95']))
        values['fdpr-0.95_std'] = np.std(np.array(values['fdpr-0.95']))

        print(f"D_out - {dataset}:\t {values['auroc_mean']:.1f} \pm {values['auroc_std']:.1f} & {values['aupr_mean']:.1f} \pm {values['aupr_std']:.1f} & {values['fdpr-0.95_mean']:.1f} \pm {values['fdpr-0.95_std']:.1f}")

####################################################################################################################################################

if __name__ == "__main__":

    # which classifier and how many recursions to use
    # which_classifier = "20_mlp"
    # which_classifier = "8_mlp"
    # which_classifier = "5_mlp"
    # which_classifier = "4_mlp"
    # which_classifier = "3_mlp"
    # which_classifier = "2_mlp"
    # which_classifier = "1_mlp"
    # which_classifier = "0_mlp"
    # which_classifier = "mlp"
    # which_classifier = "8_linear"
    # which_classifier = "5_linear"
    # which_classifier = "4_linear"
    # which_classifier = "3_linear"
    # which_classifier = "2_linear"
    # which_classifier = "1_linear"
    # which_classifier = "0_linear"

    # whether the dropout mask was fixed for all recursions
    # fix_mask = "True"
    # fix_mask = "False"

    experiments = [

    ]

    get_row(experiments, which_classifier, fix_mask)