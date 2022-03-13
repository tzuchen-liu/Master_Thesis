# Master Thesis of Tzu-Chen Liu (Tzuchen)

- Topic: Dimension Reduction of Variable Time Period Datasets of a Pneumatic System with Deep Unsupervised Learning
- Duration: From Aug. 9th 2021 to Dec. 9 2021
- Credits: 20 ECTS
- Supervisor: Faras Brumand

## Overview

- Applying autoencoder for dimension reduction on a simulation dataset from a pneumatic machine model.
- Classifying leakage type as an application of condition monitoring. 
- Using python3 as programming language.

## Documentation 

The original thesis pdf, the final presentation powerpoint, original plots for the thesis and presentation, and the result plottings are included in this repository.

## Install and run

1. Use the package manager [pip](https://pip.pypa.io/en/stable/) 
   to install [virtualenv](https://virtualenv.pypa.io/en/stable/):
```
pip3 install virtualenv
```
2. Create a virtual environment:

```
virtualenv -p {path to python3} {name of virtual environment}
```
Note: You can check out the Python3 path on your device e.g. with 
``which python3.8``.

3. Activate the virtual environment:
```
source {name of virtual environment}/bin/activate
```

4. After activating your virtual environment, install the 
   requirements of this project:
```
pip3 install -r requirements.txt
```

## Usage of the code

1. app/preprocessing.py:

This file is used to preprocess the raw data, to create a clean and normalized dataset to train autoencoder models or classification models.

2. app/train.py:

This file is used to train autoencoder models.

3. app/latent_analysis.py:

This file contains methods to analyze the latent space. 

4. app/classification.py:

This file is used to train leakage classification models. 

5. app/visualization.py:

This file contains methods to visualize training related values and results.

6. app/machine_learning/ae.py:

This file contains the autoencoder model and its encoder and decoder.

7. app/helper_func/utils.py:

This file contains methods that are used by multiple of above mentioned scripts, or that have the potential to be used outside of this project.

8. app/helper_func/config.py:

This file contains configuration settings of above mentioned scripts.
<br />
<br />
<br />
These scripts can be run by:
```
python {path to file} + {filename}
```
E.g. ``python app/train.py``
