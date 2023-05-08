import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing import image

import numpy as np
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class NeurofluxModel:
    def __init__(self, model_type='pretrained', save_path=None):
        self.model_type = model_type
        self.save_path = save_path
        self.model = None
        self.history = None
        self.CLASS_NAMES = ['EO', 'IO', 'IPTE', 'LO', 'PTE']
        
        if self.model_type == 'pretrained':
            self.model = self._get_pretrained_model()
        elif self.model_type == 'scratch':
            self.model = self._get_scratch_model()
    
    def _get_pretrained_model(self):
        base_model = ResNet152V2(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
        for layer in base_model.layers[:-15]:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(0.3)(x)
        output  = Dense(units=len(self.CLASS_NAMES), activation='softmax')(x)
        model = Model(base_model.input, output)
                
        return model
    
    def _get_scratch_model(self):
        inputs = Input(shape=(256, 256, 3))
        x = MaxPooling2D((2, 2))(inputs)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(len(self.CLASS_NAMES), activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    def train(self, train_data, val_data, epochs=10, batch_size=32, learning_rate= 0.001):
    
        # To handle data imbalance
        class_weights = {
            0: 1.0,  # class 1
            1: 3.0,  # class 2
            2: 2.0,  # class 3
            3: 1.5,  # class 4
            4: 1.0   # class 5
        }

        loss = CategoricalCrossentropy(class_weights)
        optimizerLR = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizerLR, 
                            loss=loss, 
                            metrics=['accuracy', 'Precision', 'Recall'])
        self.history = self.model.fit(train_data, 
                                        epochs=epochs, 
                                        validation_data=val_data, 
                                        batch_size=batch_size,
                                        callbacks=[EarlyStopping(patience=3)])
        
        if self.save_path is not None:
            self.model.save(self.save_path)
    
    def evaluate(self, test_data):
        results = self.model.evaluate(test_data, return_dict=True)
        loss = results['loss']
        accuracy = results['accuracy']
        print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

    def predict(self, image_path):
        img = image.load_img(image_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.
        prediction = self.model.predict(img_array)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class = self.CLASS_NAMES[predicted_class_index]
        return predicted_class
    
    def predict_batch(self, image_paths):
        predictions = []
        for image_path in image_paths:
            predictions.append(self.predict(image_path))
            
        return predictions
    
    