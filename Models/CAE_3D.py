# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:26:14 2018

@author: Saeed Mhq
"""

# In the name of GOD
#---------------------

import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1.):
    K = tf.keras.backend
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

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

def Encoder(img_size, latent_dim):
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
    model = tf.keras.Model(inLayer, x)
    return model
    
def Decoder(latent_dim):
    inLayer = tf.keras.layers.Input(shape=(latent_dim,))
#    x = tf.keras.layers.Dense(128, activation='relu')(inLayer)
    x = tf.keras.layers.Dense(512, activation='relu')(inLayer)
    x = tf.keras.layers.Reshape((8,8,8,1))(x)
    x = UpConvBlock(x, filters=64, kernel=(4,4,4), strides=(2,2,2))
    x = ConvBlock(x, filters=64, kernel=(3,3,3), strides=(1,1,1))
    x = UpConvBlock(x, filters=32, kernel=(4,4,4), strides=(2,2,2))
    x = ConvBlock(x, filters=32, kernel=(3,3,3), strides=(1,1,1))
    x = UpConvBlock(x, filters=16, kernel=(4,4,4), strides=(2,2,2))
    x = ConvBlock(x, filters=16, kernel=(3,3,3), strides=(1,1,1))
    x = UpConvBlock(x, filters=16, kernel=(4,4,4), strides=(2,2,2))
    x = tf.keras.layers.Conv3D(filters=1, kernel_size=(3,3,3), strides=(1,1,1),
                               padding='same', activation=None)(x)
    model = tf.keras.Model(inLayer, x)
    return model

if __name__ == '__main__':
    img_size = (128,128,128,1)
    latent_dim = 128
    encoder = Encoder(img_size, latent_dim)
    decoder = Decoder(latent_dim)
    inLayer = tf.keras.layers.Input(shape=img_size)
    CAE_3D = tf.keras.Model(inLayer,decoder(encoder(inLayer)))
    CAE_3D.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss = dice_coef_loss, metrics = ['accuracy'])
    CAE_3D.summary()
    
    tf.keras.utils.plot_model(CAE_3D, to_file='./CAE_Model.png', show_shapes=True)
    tf.keras.utils.plot_model(encoder, to_file='./CAE_Encoder.png', show_shapes=True)
    tf.keras.utils.plot_model(decoder, to_file='./CAE_Decoder.png', show_shapes=True)