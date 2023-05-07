import os 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# get class names
CLASS_NAMES = ['EO', 'IO', 'IPTE', 'LO', 'PTE']

# Oversample minority classes
class_weights = {0: 1, 1: 3, 2: 2, 3: 1, 4: 1}

# some files had jpg in their extention and had some problems to upload so decided to change them to jpeg
def change_file_extensions(data_dir, new_ext):
    # Loop through all the subdirectories (train, test, validation)
    for sub_dir in os.listdir(data_dir):
        sub_dir_path = os.path.join(data_dir, sub_dir)

        # Loop through all the classes (class1, class2, etc.)
        for class_dir in os.listdir(sub_dir_path):
            class_dir_path = os.path.join(sub_dir_path, class_dir)

            # Loop through all the images in the class directory
            # loop through all files in the directory
            for filename in os.listdir(class_dir_path):
                # set the path of the file
                file_path = os.path.join(class_dir_path, filename)

                # get the file name and extension
                file_name, file_ext = os.path.splitext(file_path)


                # check if the file extension is ".jpg"
                if file_ext != new_ext:
                    # rename the file with the new extension
                    os.rename(file_path, file_name + new_ext)
                    print("File extension changed successfully.")
                    

# data augmentation
def augment_data(data_dir, batch_size=32, img_height=256, img_width=256): #, save_dir
    
    SEED = 1234
    tf.random.set_seed(SEED)  
    # get class names
    CLASS_NAMES = ['EO', 'IO', 'IPTE', 'LO', 'PTE']

    # create a data generator
    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_data_path = os.path.join(data_dir, 'train')
    train_generator = data_gen.flow_from_directory(
        train_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes=CLASS_NAMES,
        seed=SEED,
        shuffle=True
    )
    
    val_data_path = os.path.join(data_dir, 'validation')
    val_generator = data_gen.flow_from_directory(
        val_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator

