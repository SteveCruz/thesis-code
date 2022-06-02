##############################################################################################################################################################
##############################################################################################################################################################
"""
Create the entropy plots used in the paper. Simply provide the experiment for which you want to create it.
Then select which classifier and recursion you want to use.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import toml
import numpy as np

from scipy import stats

import eval_cae

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("pdf")
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif', serif='Times New Roman')
plt.rc('axes', labelsize=13)
plt.rc('font', size=13)  
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('savefig', dpi=300) 

# width as measured in inkscape
PLT_WIDTH = 6.053 # full width image for thesis
PLT_HEIGHT = PLT_WIDTH / 1.618

##############################################################################################################################################################
##############################################################################################################################################################

def forward(x):
    return x**(1/2)

def inverse(x):
    return x**2

def plot_entropy(experiment, which_classifier):

    # get the path to all the folder and files
    folders = eval_cae.get_folder(experiment)

    train_config = toml.load(folders["config"])

    # get all the performance files in the data folder
    files = list(folders["data-entropy"].glob("*.npy"))

    results_ood = dict()
    results_eval = dict()

    # for each performance file
    for file in sorted(files):

        # make sure to use only 2 recursions
        if "recursion" in file.name:
            if not which_classifier in file.name:
                continue

        what_is_it = file.name.split("_")[0]
        ood_dataset_name = file.name.split("eval_")[-1].split("_")[0].split(".npy")[0]
        eval_dataset_name = file.name.split("train_")[-1].split("_")[0].split(".npy")[0]

        # avoid using itself as ood
        if ood_dataset_name == train_config["dataset"]["name"]:
            continue

        if what_is_it.lower() != "ood":
            results_eval[ood_dataset_name] = np.load(file)
        else:
            results_ood[ood_dataset_name] = np.load(file)

    fig, ax = plt.subplots()

    plt.hist(list(results_eval.values())[0], bins=50, histtype='stepfilled', linewidth=1.0, alpha=0.75, density=False, label=f"$D_{{in}}$ - {eval_dataset_name.upper()}")

    for name, data in results_ood.items():
        plt.hist(data, bins=50, histtype='step', linewidth=1.0, alpha=0.99, density=False, label=f"$D_{{out}}$ - {name.upper()}")

    
    # non-linear scaling for y-axis
    ax.set_yscale('function', functions=(forward, inverse))

    plt.xlim([0, 1.0])
    plt.ylim([0, 1450])

    plt.legend()
    plt.grid(color="0.9", linestyle='-', linewidth=1)
    plt.box(False)
    plt.xlabel("Entropy")

    fig.subplots_adjust(left=.09, bottom=.14, right=.97, top=.98)
    fig.set_size_inches(PLT_WIDTH, PLT_HEIGHT)
    fig.savefig(folders["images"].parent / "entropy.png")
    fig.savefig(folders["images"].parent / "entropy.pdf")
    plt.close(fig=fig)


    cifar_results = results_ood["cifar10"]
    test_results = list(results_eval.values())[0]
    cifar_all_distances = 0
    test_all_distances = 0
    for x,y in results_ood.items():
        if x == "test":
            continue
        cifar_all_distances += stats.wasserstein_distance(cifar_results, y) 
        test_all_distances += stats.wasserstein_distance(test_results, y) 

    print(f"Sum of distances between test and all D_out: {test_all_distances:.3f}")
    print(f"Sum of distances between Cifar10 and all other D_out: {cifar_all_distances:.3f}")

    return test_all_distances, cifar_all_distances

##############################################################################################################################################################
##############################################################################################################################################################

if __name__ == "__main__":

    # which classifier and how many recursions to use
    # nbr_recursions_classifier
    # which_classifier = "20_mlp"
    # which_classifier = "5_mlp"
    # which_classifier = "4_mlp"
    # which_classifier = "3_mlp"
    # which_classifier = "2_mlp"
    # which_classifier = "1_mlp"
    # which_classifier = "0_mlp"
    # which_classifier = "mlp"
    # which_classifier = "20_knn"
    # which_classifier = "10_knn"
    # which_classifier = "5_knn"
    # which_classifier = "4_knn"
    # which_classifier = "3_knn"
    # which_classifier = "2_knn"
    # which_classifier = "1_knn"
    # which_classifier = "0_knn"

    all_test_distances = []
    all_cifar_distances = []

    experiments = [

    ]

    for experiment in experiments:
        print("-" * 37)
        print("Experiment: ", str(experiment))
        test_distances, cifar_distances = plot_entropy(experiment, which_classifier)
        all_test_distances.append(test_distances)
        all_cifar_distances.append(cifar_distances)

    print()
    print("*" * 77)
    print("*" * 77)
    print(f"Mean of distances between test and all D_out: {np.mean(np.array(all_test_distances)):.3f}  +- {np.std(np.array(all_test_distances)):.3f}")
    print(f"Mean of distances between Cifar10 and all other D_out: {np.mean(np.array(all_cifar_distances)):.3f} +- {np.std(np.array(all_cifar_distances)):.3f}")