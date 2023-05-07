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
    
    # paths
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
    model = NeurofluxModel(model_type='pretrained', save_path='./models/ResNet152V2.h5')
    model.train(train_data, validation_data, epochs=10, batch_size=32, learning_rate=1e-3)
    model.evaluate(test_data)
    
    plot_history_epoch_accuracy(model.history)
    
    