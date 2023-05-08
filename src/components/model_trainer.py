# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.append('/home/amine/Desktop/test_tech')


# Import local resources
from src.components.model_class import NeurofluxModel 
from src.components.data_transformation import augment_data
from src.utils import plot_history_epoch_accuracy

import tensorflow as tf 
from tensorflow.keras.preprocessing import image_dataset_from_directory





if __name__ == '__main__':
    
    # pathsfrom_logits
    data_dir = '/home/amine/Desktop/test_tech/data_split'
    test_data_path = '/home/amine/Desktop/test_tech/data_split/test'
    
    test_data = image_dataset_from_directory(
        test_data_path,
        labels='inferred',
        label_mode='categorical',
        image_size=(256, 256),
        batch_size=32
    )
    
    train_data, validation_data = augment_data(data_dir, batch_size=32, img_height=256, img_width=256)
    
    print('Model Trainer')
    
    # get the model from the model_class and train the model using the model class
    model = NeurofluxModel(model_type='scratch', save_path='./models/scratch_model.h5') # change it to scratch if you want to train the scratch model
    
    model.train(train_data, validation_data, epochs=20, batch_size=32, learning_rate=1e-3)
    model.evaluate(test_data)
    print(model.model.summary())
    plot_history_epoch_accuracy(model.history)
    
    # test prediction
    test_image_path = "" # give a path to an image
    predict_image = model.predict(test_image_path) 
    print("the predicted class is: ", predict_image)
    
    
    