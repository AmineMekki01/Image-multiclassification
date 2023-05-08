# test_tech : End To End Deep Laerning image multi classification Project.

 ? The implementations are inside the src folder?

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
        
### 4- Adding flask app and docker file .


I think if i had more time for this task i could hav eexplored more ways to analyse it and push the models to the limit and try GAN (altough that there are some limitation to use in term of sample size, but there are some work arounds that can be done) 
I didn;t have time to fully test the GAN to its limits. I only tested that it works using small number of epochs and steps.

 ## Quick explanation 
 We can see that we are not getting musch of accuracies, precision and recall. we have little amount of data that it is imbalanced. but still our pretrained model performs well. But still limited because of data.
 
 
 ## Model explanation : 
The class NeurofluxModel is an implementation of a deep neural network for image classification using TensorFlow and Keras libraries. The class has methods to create and train the model, evaluate its performance, and make predictions on new images.

The model can be either pretrained or trained from scratch, and the user can specify the model type when creating an instance of the class. The pretrained model is based on the ResNet152V2 architecture and is initialized with weights pre-trained on the ImageNet dataset. 
The scratch model is a simpler architecture with convolutional and pooling layers followed by dense layers.

- The train method trains the model using the specified training and validation data. The loss function used is the categorical cross-entropy, and the optimizer used is Adam with a specified learning rate. The method also handles class imbalance by specifying class weights. Early stopping is used as a callback to prevent overfitting.

- The evaluate method evaluates the performance of the trained model on the specified test data and prints the test loss and accuracy.

- The predict method takes the path to a single image as input and predicts its class using the trained model. 
- The predict_batch method takes a list of image paths as input and predicts the classes of all images in the list using the predict method. The predicted classes are returned as a list.








