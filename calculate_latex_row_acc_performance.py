####################################################################################################################################################
####################################################################################################################################################
"""
Compute performances accuracies for several seed runs.
Make sure to provide 10 runs of the same kind of experiments.
Uncomment the .performance file you want to consider.
"""
####################################################################################################################################################
####################################################################################################################################################

import toml
import numpy as np
from pathlib import Path

####################################################################################################################################################
####################################################################################################################################################

def get_row(experiment):

    assert len(experiments) == 10

    old_train = ""

    # keep track
    all_results = []

    for experiment in experiments:

        # get the config file to get the training data
        config_file = Path("../results") / experiment / "cfg.toml"
        config = toml.load(config_file)
        train = config["dataset"]["name"]

        if old_train != "":
            if train != old_train:
                raise ValueError("Not all models are from the same dataset")

        # highlight the one you want to use for the evaluation.
        # result_file = Path("results") / experiment / "logs" / "test_accuracy_linear.performance"
        # result_file = Path("results") / experiment / "logs" / "test_accuracy_orss-Sharan-all_mlp.performance"
        # result_file = Path("results") / experiment / "logs" / "test_accuracy_orss-Sharan-all_knn-1.performance"
        # result_file = Path("results") / experiment / "logs" / "test_accuracy_orss-Sharan-test_linear.performance"
        # result_file = Path("results") / experiment / "logs" / "test_accuracy_orss-Sharan-test_knn-1.performance"
        # result_file = Path("results") / experiment / "logs" / "test_accuracy_orss-Sharan-test_mlp.performance"
        # result_file = Path("results") / experiment / "logs" / "test_accuracy_orss-X5-test_linear.performance"
        # result_file = Path("results") / experiment / "logs" / "test_accuracy_orss-X5-test_knn-1.performance"
        # result_file = Path("results") / experiment / "logs" / "test_accuracy_orss-X5-test_mlp.performance"
        # result_file = Path("results") / experiment / "logs" / "orss-all-sharan.accuracy"
        # result_file = Path("results") / experiment / "logs" / "orss-test-x5.accuracy"
        # result_file = Path("results") / experiment / "logs" / "orss-test-sharan.accuracy"

        # get the value
        with result_file.open("r") as f:
            value = float(f.readlines()[0])*100

        # keep track of the value
        all_results.append(value)

        old_train = train


    all_results = np.array(all_results)
    mean_results = np.mean(all_results)
    std_results = np.std(all_results)

    print("-"*77)
    print(experiment)
    print(f"Training: \t\t {train}")
    print("-"*77)
    if "metric" in config["model"]:
        print(f'Metric: {config["model"]["metric"]} - Impossible: {config["training"]["make_instance_impossible"]}')
        print("-"*77)
    print(f"{mean_results:.1f} \pm {std_results:.1f}")

####################################################################################################################################################

if __name__ == "__main__":

    experiments = [



        ]

    get_row(experiments)