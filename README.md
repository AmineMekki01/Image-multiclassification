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

### 3- Creating general architecture of the code and implementation.
  - src : our package folder where i will be implementing everything.
    - components : 
        - data_augmentation_GAN : in which i was trying to implement gan and train it (i don't think i will do it, it takes a lot of time and i have other things to do : internship and my own project.)       
                - GAN_model.py : GAN implemtation
                - GAN_trainer : GAN Traning (need to finish testing to clean it)
        - data_splitter.py : to split the data into 70% train, 15% validation and 15% test.
        - data_transformation.py : where i did a simple data augmentation using tensorflow libraries.
        - model_class.py : the module where i defined my model class of the pretrained model and scratch model.
        
    - pipeline : TO use for flask app if i have time.
    
   
        - model_trainer : the file used to train the models.
        
