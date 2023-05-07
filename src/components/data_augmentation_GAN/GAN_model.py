from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Reshape, Input, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


class GenerativeAdversarialNetwork:

    def __init__(self):

        self.NOISE_DIM = 100  
        self.BATCH_SIZE = 4 
        self.STEPS_PER_EPOCH = 3000
        self.EPOCHS = 50
        self.SEED = 40
        self.WIDTH, self.HEIGHT, self.CHANNELS = 256, 256, 1
        self.OPTIMIZER = Adam(0.0002, 0.5)

    def build_generator(self):

        model = Sequential([

            Dense(64*64*256, input_dim=self.NOISE_DIM),
            LeakyReLU(alpha=0.2),
            Reshape((64, 64, 256)),
            
            Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Conv2D(self.CHANNELS, (4, 4), padding='same', activation='tanh')
        ], 
        name="generator")
        model.compile(loss="binary_crossentropy", optimizer=self.OPTIMIZER)

        return model

    def build_discriminator(self):
        
        model = Sequential([

            Conv2D(64, (3, 3), padding='same', input_shape=(self.WIDTH, self.HEIGHT, self.CHANNELS)),
            LeakyReLU(alpha=0.2),

            Conv2D(128, (3, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            Conv2D(128, (3, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.2),
            
            Conv2D(256, (3, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.2),

            
            Flatten(),
            Dropout(0.4),
            Dense(1, activation="sigmoid", input_shape=(self.WIDTH, self.HEIGHT, self.CHANNELS))
        ], name="discriminator")
        model.compile(loss="binary_crossentropy",
                            optimizer=self.OPTIMIZER)

        return model

    def generative_adversarial_network(self, discriminator, generator):
        discriminator.trainable = False 

        gan_input = Input(shape=(self.NOISE_DIM,))
        fake_image = generator(gan_input)

        gan_output = discriminator(fake_image)

        gan = Model(gan_input, gan_output, name="gan_model")
        gan.compile(loss="binary_crossentropy", optimizer=self.OPTIMIZER)

        return gan, gan.summary()
    
    
    