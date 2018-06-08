# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:57:08 2018

@author: Saeed Mhq
"""

# In the name of GOD
#---------------------

import tensorflow as tf
import keras.backend as K

def ConvBlock(x, filters, kernel, strides):
    x = tf.keras.layers.Conv3D(filters, kernel, strides,
                               padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

def UpConvBlock(x, filters, kernel, strides):
    x = tf.keras.layers.Conv3DTranspose(filters, kernel, strides,
                                        padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

def Generator(latent_dim):
    generator_input = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(512, activation='relu')(generator_input)
    x = tf.keras.layers.Reshape((8,8,8,1))(x)
    x = UpConvBlock(x, filters=64, kernel=(4,4,4), strides=(2,2,2))
    x = ConvBlock(x, filters=64, kernel=(3,3,3), strides=(1,1,1))
    x = UpConvBlock(x, filters=32, kernel=(4,4,4), strides=(2,2,2))
    x = ConvBlock(x, filters=32, kernel=(3,3,3), strides=(1,1,1))
    x = UpConvBlock(x, filters=16, kernel=(4,4,4), strides=(2,2,2))
    x = ConvBlock(x, filters=16, kernel=(3,3,3), strides=(1,1,1))
    x = UpConvBlock(x, filters=16, kernel=(4,4,4), strides=(2,2,2))
    x_generated = tf.keras.layers.Conv3D(filters=1, kernel_size=(3,3,3), strides=(1,1,1),
                               padding='same', activation='sigmoid')(x)
    generator = tf.keras.Model(generator_input, x_generated)
    return generator
    
def CVAE(img_size, batch_size, latent_dim):
    inLayer = tf.keras.layers.Input(shape=img_size)
    x = ConvBlock(inLayer, filters=16, kernel=(3,3,3), strides=(2,2,2))
    x = ConvBlock(x, filters=16, kernel=(3,3,3), strides=(1,1,1))
    x = ConvBlock(x, filters=32, kernel=(3,3,3), strides=(2,2,2))
    x = ConvBlock(x, filters=32, kernel=(3,3,3), strides=(1,1,1))
    x = ConvBlock(x, filters=64, kernel=(3,3,3), strides=(2,2,2))
    x = ConvBlock(x, filters=64, kernel=(3,3,3), strides=(1,1,1))
    x = ConvBlock(x, filters=64, kernel=(3,3,3), strides=(2,2,2))
    x = ConvBlock(x, filters=1, kernel=(1,1,1), strides=(1,1,1))
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim, activation=None)(x)
    drp = tf.keras.layers.Dropout(rate=0.5)(x)
    z_mean = tf.keras.layers.Dense(latent_dim)(drp)
    z_log_var = tf.keras.layers.Dense(latent_dim)(drp)
    
    epsilon_std = 1.0
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon
    
    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = tf.keras.Model(inLayer, z_mean)
 
    generator = Generator(latent_dim)
    x_decoded = generator(z)
    
    generator_input = tf.keras.layers.Input(shape=(latent_dim,))
    generator = tf.keras.Model(generator_input, generator(generator_input))

    
    class CustomVariationalLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)
    
        def vae_loss(self, x, x_decoded):
            x = K.flatten(x)
            x_decoded = K.flatten(x_decoded)
            xent_loss = img_size[0] * img_size[1] * img_size[2] * tf.keras.metrics.binary_crossentropy(x, x_decoded)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)
    
        def call(self, inputs):
            x = inputs[0]
            x_decoded = inputs[1]
            loss = self.vae_loss(x, x_decoded)
            self.add_loss(loss, inputs=inputs)
            # We don't use this output.
            return x
        
#    y = CustomVariationalLayer()([inLayer, Generator(latent_dim)(z)])
    y = CustomVariationalLayer()([inLayer, x_decoded])

    CVAE_3D = tf.keras.Model(inLayer, y)

    return encoder, generator, CVAE_3D
    
if __name__ == '__main__':
    img_size = (128,128,128,1)
    batch_size = 1
    latent_dim = 128
    encoder, generator, CVAE_3D = CVAE(img_size, batch_size, latent_dim)
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    CVAE_3D.compile(optimizer=opt, loss=None)
    CVAE_3D.summary()
    tf.keras.utils.plot_model(CVAE_3D, to_file='CVAE_Model.png', show_shapes=True)
    tf.keras.utils.plot_model(encoder, to_file='CVAE_Encoder.png', show_shapes=True)
    tf.keras.utils.plot_model(generator, to_file='CVAE_Generator.png', show_shapes=True)

