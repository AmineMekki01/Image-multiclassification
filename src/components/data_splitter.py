from sklearn.model_selection import train_test_split
import os
import shutil

def split_data(source, destination, train_size):
    """
    Split the data into train, validation and test sets.
    """
    classes = os.listdir(source)
    
    create_directories(source, destination)
    
    for cls in classes:
        cls_folder = os.path.join(source, cls)
        train_class_folder = os.path.join(destination + "train", cls)
        val_class_folder = os.path.join(destination + "validation", cls)
        test_class_folder = os.path.join(destination + "test", cls)

        # Get a list of all the image files in the class folder
        files = os.listdir(cls_folder)
        images = [os.path.join(cls_folder, f) for f in files if f.endswith('.jpg') or f.endswith('.png')]

        # Split the images into train, validation, and test sets
        train_images, test_images = train_test_split(images, train_size=train_size, random_state=42)
        val_images, test_images = train_test_split(test_images, train_size=0.5, random_state=42)

        # Copy the images to the corresponding folders
        for image in train_images:
            shutil.copy(image, val_class_folder)

        for image in val_images:
            shutil.copy(image, val_class_folder)

        for image in test_images:
            shutil.copy(image, test_class_folder)

def create_directories(source_destination, destination_folder):
    
    """
    Create the train, validation and test directories if they do not exist.
    """
    
    if not os.path.exists(os.path.join(destination_folder)):
        os.makedirs(os.path.join(destination_folder))
        
    classes = os.listdir(source_destination)
    
    if not os.path.exists(os.path.join(destination_folder, "train")):
        os.makedirs(os.path.join(destination_folder, "train"))
                    
    if not os.path.exists(os.path.join(destination_folder, "validation")):
        os.makedirs(os.path.join(destination_folder, "validation"))
    
    if not os.path.exists(os.path.join(destination_folder, "test")):
        os.makedirs(os.path.join(destination_folder, "test"))
        

    for imClass in classes:
        if not os.path.exists(os.path.join(destination_folder + "/train/", imClass)):
            os.makedirs(os.path.join(destination_folder + "/train/", imClass))
            
            
        if not os.path.exists(os.path.join(destination_folder + "/validation/", imClass)):
            os.makedirs(os.path.join(destination_folder + "/validation/", imClass))
            
        if not os.path.exists(os.path.join(destination_folder + "/test/", imClass)):
            os.makedirs(os.path.join(destination_folder + "/test/", imClass))


if __name__ == "__main__":

    source_folder = '/home/amine/Desktop/test_tech/data/'
    destination_folder = '/home/amine/Desktop/test_tech/data_split_test/'
    train_size = 0.7
        
    split_data(source_folder, destination_folder, train_size)
