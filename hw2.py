import numpy as np
from tensorflow.keras.datasets import cifar10

(X_train, Y_train), (_, _) = cifar10.load_data()

# Get automobile data
dataset = []
for i in range(len(X_train)):
    if Y_train[i] == [9]:  # 1: automobile
        dataset.append(X_train[i])

trainset = np.array(dataset)
in_shape = dataset[0].shape

from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Define generator
G = Sequential()

# foundation for 4x4 image
G.add(Dense(256 * 4 * 4, input_dim=100))
G.add(LeakyReLU(alpha=0.2))
G.add(Reshape((4, 4, 256)))

# upsample to 8x8
G.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
G.add(LeakyReLU(alpha=0.2))

# upsample to 16x16
G.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
G.add(LeakyReLU(alpha=0.2))

# upsample to 32x32
G.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
G.add(LeakyReLU(alpha=0.2))

# output layer
G.add(Conv2D(3, (3,3), activation='tanh', padding='same'))

# Define discriminator
D = Sequential()

# normal
D.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
D.add(LeakyReLU(alpha=0.2))

# downsample
D.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
D.add(LeakyReLU(alpha=0.2))

# downsample
D.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
D.add(LeakyReLU(alpha=0.2))

# downsample
D.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
D.add(LeakyReLU(alpha=0.2))

# classifier
D.add(Flatten())
D.add(Dropout(0.4))
D.add(Dense(1, activation='sigmoid'))

# compile model
D.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

# Define GAN
def define_gan(g_model, d_model):

    # GAN is training Generator by the loss of Disciminator, make weights in the discriminator not trainable
    d_model.trainable = False

    model = Sequential()

    # concatenate generator and discriminator
    model.add(g_model)
    model.add(d_model)

    return model

# build GAN
GAN = define_gan(G, D)

# compile model
GAN.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

import math
import tensorflow as tf
# Configs
max_epoch = 200
batch_size = 32

half_batch = int(batch_size/2)
# Trai
with tf.device("/device:GPU:0"):
  for epoch in range(max_epoch):

      for i in range(math.ceil(len(trainset) / half_batch)):

        # Update discriminator by real samples
          r_images = trainset[i*half_batch:(i+1)*half_batch]
          d_loss_r, _ = D.train_on_batch(r_images, np.ones((len(r_images), 1)))

        # Update discriminator by fake samples
          f_images = G.predict(np.random.normal(0, 1, (half_batch, 100))) # generate fake images
          d_loss_f, _ = D.train_on_batch(f_images, np.zeros((len(f_images), 1)))

          d_loss = (d_loss_r + d_loss_f)/2

        # Update generator
          g_loss = GAN.train_on_batch(np.random.normal(0, 1, (batch_size, 100)), np.ones((batch_size, 1)))

        # Print training progress
          print(f'[Epoch {epoch+1}, {min((i+1)*half_batch, len(trainset))}/{len(trainset)}] D_loss: {d_loss:0.4f}, G_loss: {g_loss:0.4f}')

    # Print validation result
    # evaluate discriminator on real examples
      _, acc_real = D.evaluate(trainset, np.ones((len(trainset), 1)), verbose=0)

    # evaluate discriminator on fake examples
      f_images = G.predict(np.random.normal(0, 1, (len(trainset), 100)))
      _, acc_fake = D.evaluate(f_images, np.zeros((len(trainset), 1)), verbose=0)

    # summarize discriminator performance
      print(f'[Epoch {epoch}] Accuracy real: {acc_real*100}, fake: {acc_fake*100}')

from tensorflow.keras.models import save_model

save_model(G, 'g.h5')

# for resume training
D.trainable = True
save_model(D, 'd.h5')

D.trainable = False
save_model(GAN, 'gan.h5')
