import matplotlib.pyplot as plt

def plot_history_epoch_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()
    
    
def check_image_extension(image_path):
    if image_path[-4:] != 'jpeg':
        image_path = image_path[:-4] + 'JPEG'
    return image_path