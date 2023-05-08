import numpy as np 
import os 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm


from GAN_model import GenerativeAdversarialNetwork


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()


print("REPLICAS: ", strategy.num_replicas_in_sync)

train_dir = '/home/amine/Desktop/test_tech/data_split/train/IO'

# Instantiate the Model
gan = GenerativeAdversarialNetwork()

generator = gan.build_generator()
discriminator = gan.build_discriminator()
gan_model, gan_model_summary = gan.generative_adversarial_network(discriminator=discriminator, generator=generator)
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))


print(gan_model_summary)

# Loading the Image data
def load_images(folder):
    
    imgs = []
    target = 1
    labels = []
    for i in os.listdir(folder):
        img_dir = os.path.join(folder,i)
        try:
            img = cv2.imread(img_dir)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (256,256))
            imgs.append(img)
            labels.append(target)
        except:
            continue
        
    imgs = np.array(imgs)
    labels = np.array(labels)
    
    return imgs, labels


data_train_images, data_train_labels = load_images(train_dir)

print(f"shape is {data_train_images.shape} and labels is {data_train_labels.shape}")

# get 20 random numbers to get 20 random images
idxs = np.random.randint(0, data_train_images.shape[0], 20)
X_train = data_train_images[idxs]

# Normalize the Images
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Reshape images 
X_train = X_train.reshape(-1, gan.WIDTH, gan.HEIGHT, gan.CHANNELS)


def save_images(noise, save=False):
    generated_images = generator.predict(noise)
    
    # save the images i current folder
    if save:
        for i in range(generated_images.shape[0]):
            plt.imsave(f"generated_images_{i}.png", generated_images[i, :, :, 0], cmap="gray")
            
            

print("training started")

# Training Process
np.random.seed(gan.SEED)
g_loss_std = 10
d_loss_std = 10
inception_score_std = 10

for epoch in range(gan.EPOCHS):
    for batch in range(gan.STEPS_PER_EPOCH):

        # Generating the noise and creating fake images
        noise = np.random.normal(0,1, size=(gan.BATCH_SIZE, gan.NOISE_DIM))
        fake_X = generator.predict(noise)
        
        # Get real images from the data
        idx = np.random.randint(0, X_train.shape[0], size=gan.BATCH_SIZE)
        real_X = X_train[idx]

        # Put them together
        X = np.concatenate((real_X, fake_X))

        # Creating the labels
        disc_y = np.zeros(2*gan.BATCH_SIZE)
        disc_y[:gan.BATCH_SIZE] = 1

        # One step of gradient update on discriminator
        d_loss = discriminator.train_on_batch(X, disc_y)
        
        # Create all labels as real to fool the discriminator
        # One step of gradient update on the entire gan_model
        # Note: No changes on discriminator here, only Generator weights update 
        y_gen = np.ones(gan.BATCH_SIZE)
        g_loss = gan_model.train_on_batch(noise, y_gen)
        
        real_X = np.repeat(real_X, 3, axis=-1)
        fake_X = np.repeat(fake_X, 3, axis=-1)
        
        real_images = preprocess_input(real_X)
        fake_images = preprocess_input(fake_X)
        
        # Extract features from real and fake images using Inception v3
        real_features = inception_model.predict(real_images)
        fake_features = inception_model.predict(fake_images)

        # Calculate mean and covariance of feature vectors for real and fake images
        mu_real, cov_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_fake, cov_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

        # Calculate FID score
        diff = mu_real - mu_fake
        sq_diff = diff.dot(diff)
        
        covmean = sqrtm(cov_real.dot(cov_fake))
        
        
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        trace_sqrt = np.trace(cov_real + cov_fake - 2.0*covmean)
        
        inception_score = sq_diff + trace_sqrt
        
        print(type(inception_score))
        
        

    # Check Progress at every Epoch
    print(f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f} and inception score is {inception_score}")
    noise = np.random.normal(0, 1, size=(10,gan.NOISE_DIM))
    
    # if f_loss and d_loss are better that the last iteration save the image save_images(noise, save=True)
    # if g_loss < g_loss_std and d_loss < d_loss_std:
    if inception_score_std > inception_score:
        save_images(noise, save=True)
        # g_loss_std = g_loss
        # d_loss_std = d_loss
        inception_score_std = inception_score
    
# Save the trained generator model weights to a file
generator.save_weights('/home/amine/Desktop/test_tech/models/generator_weights.h5')