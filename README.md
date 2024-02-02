# Information Retrieval. Application of Deep Learning.
This repository contains my final project for the NLP course in FMI. The main objective of the project was to implement and train a neural machine translation model based on the Transformer architecture that translates from English to Bulgarian.

## Repository content
* `en_bg_data/` - folder containing the datasets for training, evaluating and testing the model
* `wandb/` - folder containing the diagrams from training, evaluating the model during training
* `parameters.py` - all parameters used for setting the the model config and its training process
* `model.py` - contains the implementation of every component of the Transformer architecture
* `transformer.py` - contains the training/evaluation/testing logic of the model
* `noam.py` - learning rate scheduler implementation
* `utils.py` - tools needed to prepare the datasets
* `dataset.py` - dataloader and Dataset abstractions that allow to process the datasets

## Local environment setup
Multiple dependencies are needed to start training the model and translate texts with the already trained one.
The only 2 dependencies that are needed to be installed prior to this step are:
* python >= 3.10
* conda

If these dependencies are already present on your system, then execute the following command:
```
conda env create -f environment.yml 
```
where `environment.yml` contains are other dependencies which are directly connected with the training of the model, such as `pytorch`.

## Dataset preparation
The following command prepares the vocabulary and datasets needed for training/evaluating/testing the model
```
python utils.py prepare
```

## Model training
```
python transformer.py train
```

## Translating a text
The ... WIP
```
python transformer.py translate <output-file-name>
```

## Bleu score testing
```
python utils.py bleu en_bg_data/test.bg <translated-file>
```
where `en_bg_data/test.bg` is the expected translation text
