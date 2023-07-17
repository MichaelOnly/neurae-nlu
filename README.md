# Neurae NLU
This repository contains code files for data generating and model researching for [Neurae](https://github.com/neurae)'s nlu module.

## Installation
For usage you need python 3.9 or higher.

Then just use ```pip install -r requirements.txt```.

## Usage

### Dataset
Use ```data_generation.py``` to generate raw data. 

Then use ```data_analysis.ipynb``` to split data on train/evaluation/test and unite them into dataset. Also it shows distribution of examples over classes and distribution of examples over example length in tokens.

### Models
To explore the optimality of finetuning parameters (lr, lr scheduler, weight decay) use one of the ```sweep``` files with matching model name.

To finetune model use ```train``` file with matching model name.

Available models
| Model      | Size |
| ---------- | :--- |
| BERT       | 110M |
| DistilBERT | 66M  |
| RoBERTa    | 125M |
| ALBERT     | 12M  |
| ELECTRA    | 110M |
| OPT        | 350M |
