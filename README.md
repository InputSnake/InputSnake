# An Exploration of The Latent Hierarchical Relationship between Medical Tasks.

The repository holds the codes of the paper An Exploration of The Latent Hierarchical Relationship between Medical Tasks.

## Installation

All the dependencies are in the requirements.txt. You can build the environment with the following command:

`conda create --name <env> --file requirements.txt`

## Data Download

The MIMIC-III data can be downloaded at: https://physionet.org/content/mimiciii/1.4/. 
This dataset is a restricted-access resource. 
To access the files, you must be a credentialed user and sign the data use agreement (DUA) for the project. 
Because of the DUA, we cannot provide the data directly.

## Instructions

To run the code in this repo, please follow the instructions below.

1. Download the MIMIC-III data.
2. Extract the features using the scripts provided at: https://github.com/MLD3/FIDDLE-experiments.
3. Use the data_process.ipynb to pre-processing the features extracted in Step 2.
3. Set the data_dir in the data_module.py to the path of the above data.
4. Set appropriate parameters in run.py.
5. Run the model by: `python run.py --model=MortInputSnake --task=mortality --mode=max --feature_selector=True`.
