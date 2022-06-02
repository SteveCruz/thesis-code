# Towards Reliable Computer Vision Feature Extraction by Novel Autoencoder Methods

This repository is the official PyTorch implementation to reproduce most of the results from my PhD thesis [Towards Reliable Computer Vision Feature Extraction by Novel Autoencoder Methods](https://sviro.dfki.de/). 

If you want to cite this work, please use the bibtex entry of my PhD thesis:

```
@article{
  TBC
}
```

## Requirements

To install the requirements:

```setup
pip3 install -r requirements.txt
```

If you encounter problems installing opencv, please consider upgrading pip first as recommended by [their FAQ](https://github.com/opencv/opencv-python#frequently-asked-questions). 

## Datasets

The datasets need to be downloaded manually.
Place the datasets inside a folder of your choice. Then define the root path to all the datasets inside `dataset.py`:
```
ROOT_DATA_DIR = Path("")
```
Potentially, you need to adapt the folder name of the downloaded datasets to match the names used inside `dataset.py`.
When you run a training script for a dataset for the first time, then the script will perform a pre-processing to center crop and resize the images and save them alongside the original images.



## Training

The hyperparameters for the different training approaches for the different models are defined in config files located in the config folder.
Modify the hyperparameters accordingly to which approach you want to use during training. 
Then, inside `train_ae.py`, `repeat_train.py`, `train_classifier` or `repeat_classifier.py` modify which config file to load, e.g.:
```
config = toml.load("cfg/dropout_cae.toml")
```
Finally, to train the model using the defined config file, run one of those commands:

```
python3 train_ae.py
```

```
python3 repeat_train_ae.py
```

`repeat_train_ae.py` will repeat the same experiment using the same hyperparameters for the different seeds defined inside the files respectively.
The config files are self-explanatory and provide the necessary information to reproduce the results. 

If you want to train (fine-tune) a (pre-trained) CNN, you can use `train_classifier.py` and `repeat_classifier.py` instead.

## Evaluation

Evaluation needs to be performed by chaining a few scripts after another.
All the experiments, which you want to evaluate, need to be located inside the results folder.

If you want to evaluate your model on all (or a subset of all) datsets, you can use `eval_ae.py` or `eval_ensemble.py`, depending on whether you want to evaluate an autoencoder approach or an ensemble of models.
The results will be saved inside the experiment folder.
Put the experiment folder names inside the script and run

```
python3 eval_ae.py
```

Afterwards, you can use a more specialiazed analysis.
For example, if you want to reproduce the entropy plot, you need to use `plot_entropy.py`

```
python3 plot_entropy.py
```

In case you want to adopt the attractor autoencoder approach, you can evaluate the model using `eval_cae.py`.

CNN can be evaluated using `eval_classifier.py`.

## Miscellaneous

Regarding the remaining scripts inside this repository, we provide some small explanations:

| Script                                 | Training dataset                                                                                  | 
|--------------------------------------- | --------------------------------------------------------------------------------------------------| 
| calculate_latex_row_acc_performance.py | Calculate the mean and standard deviation accuracy and print the latex code                       | 
| calculate_latex_row_ood_performance.py | Calculate the mean and standard deviation uncertainty (AUROC, AUPR, FPR) and print the latex code | 
| compute_complexity_dataset.py          | Compute the visual complexity of a dataset                                                        | 
| model.py                               | Autoencoder model architecture definitions                                                        | 
| pretrained_model.py                    | CNN model architecture definitions based on torchvision                                           | 
| dataset.py                             | Dataloader for the different datasets                                                             | 
| plot_entropy.py                        | Plot histogram of entropy for uncertainty estimation                                              | 
| remove_bg.py                           | Removing the background when II-PIRL is used for training                                         | 
| utils.py                               | A few helperfunctions                                                                             | 


All scripts have detailed comments and should be self-explanatory about how to use them and what their purpose is.

## Contributing

All contributions are welcome! All content in this repository is licensed under the MIT license.

## Acknowledgment

I and the work presented in this repository were supported by the Luxembourg National Research Fund (FNR) under grant number 13043281. This work was partially funded by the Luxembourg Ministry of the Economy (CVN 18/18/RED).