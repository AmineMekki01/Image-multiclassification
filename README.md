# test_tech : End To End Deep Laerning image multi classification Project.

## The task :

In this test im asked to classify five different phases of an imaginary disease called "Neuroflux disorder".

## Data

I have 5 folder with different number of images (Data imbalance!!)

## Models :

- Model1, using pretrained models (Transfer learning).
- Model2, Implement my own NN and train it from scratch.

## Implementation

### 1 - Creating a venv environement : (Good practices)

- conda create -p venv python==3.10 -y
- conda activate venv

### 2- Creating a setup.py and requirements.txt

- setup.py : to create our machine learning application as a package. Easy for deployement.

  In This file i created a function that will read all the required libraries and install them.

  Inside it there is the famous "-e ." which will trigger the installation of our package.
- requirements.txt : Inn which we will put all the used libraries.
