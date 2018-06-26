# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:26:14 2018

@author: Saeed Mhq
"""

# In the name of GOD
#---------------------

import tensorflow as tf
Lambda = tf.keras.layers.Lambda

def ConvBN(xin, filters, strides, name):
    kernel=(3,3,3)
    x = tf.keras.layers.Conv3D(filters, kernel, strides,
                               padding='same', activation=None, name=name)(xin)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

def UpConvBN(xin, filters, strides, name):
    kernel=(2,2,2)
    x = tf.keras.layers.Conv3DTranspose(filters, kernel, strides=strides,
                                        padding='same', activation=None, name=name)(xin)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

def BNConv(xin, filters, strides, name):
    kernel=(3,3,3)
    x = tf.keras.layers.BatchNormalization()(xin)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.Conv3D(filters, kernel, strides,
                               padding='same', activation=None, name=name)(x)
    return x

def BNUpConv(xin, filters, strides, name):
    kernel=(2,2,2)
    x = tf.keras.layers.BatchNormalization()(xin)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.Conv3DTranspose(filters, kernel, strides=strides,
                                        padding='same', activation=None, name=name)(x)
    return x


def get_encoder_from_CAE3D(model):
    for i,layer in enumerate(model.layers):
        if layer.name == 'encoded':
            break
    encodedIndex = i+1    
    xin = model.input
    x = model.layers[1](xin)
    for i in range(2,encodedIndex):
        x = model.layers[i](x)
    encoder = tf.keras.Model(xin, x)
    return encoder

#def get_decoder_from_CAE3D(model, latent_dim):
#    for idx,layer in enumerate(CAE3D.layers):
#        if layer.name == 'encoded':
#            break
#    encodedIndex = idx 
#    inLayer = tf.keras.layers.Input(shape=(latent_dim,), name='encoded') 
#    x = model.layers[encodedIndex+1](inLayer)
#    for i in range(encodedIndex+2,len(CAE3D.layers)):
#        x = model.layers[i](x)
#    decoder = tf.keras.Model(inLayer, x)
#    return decoder


def FullModel(img_size, latent_dim, d=2):
    inLayer = tf.keras.layers.Input(shape=img_size, name='input')
    x = ConvBN(inLayer, filters=8/d, strides=(2,2,2), name='Conv1_d')
    x = BNConv(x, filters=int(8/d), strides=(1,1,1), name='Conv2')
    x = BNConv(x, filters=int(16/d), strides=(2,2,2), name='Conv3_d')
    x = BNConv(x, filters=int(16/d), strides=(1,1,1), name='Conv4')
    x = BNConv(x, filters=int(32/d), strides=(2,2,2), name='Conv5_d')
    x = BNConv(x, filters=int(32/d), strides=(1,1,1), name='Conv6')
    x = BNConv(x, filters=int(32/d), strides=(2,2,2), name='Conv7_d')
    c = BNConv(x, filters=1, strides=(1,1,1), name='Conv8')
#    x = tf.keras.layers.Reshape((512,))(c)
    x = tf.keras.layers.Flatten(name='Flatten')(c)
    encoded = tf.keras.layers.Dense(latent_dim, activation='sigmoid', name='encoded')(x)
    
    x = tf.keras.layers.Dense(512, activation='relu', name='Dense')(encoded)
#    x = tf.keras.layers.Reshape(c.shape[1:])(x)
    x = tf.keras.layers.Reshape((8,8,8,1), name='Reshape')(x)
    x = UpConvBN(x, filters=int(32/d), strides=(2,2,2), name='UpConv1')
    x = BNConv(x, filters=int(32/d), strides=(1,1,1), name='Conv9')
    x = BNUpConv(x, filters=int(32/d), strides=(2,2,2), name='UpConv2')
    x = BNConv(x, filters=int(16/d), strides=(1,1,1), name='Conv10')
    x = BNUpConv(x, filters=int(16/d), strides=(2,2,2), name='UpConv3')
    x = BNConv(x, filters=int(8/d), strides=(1,1,1), name='Conv11')
    x = BNUpConv(x, filters=int(8/d), strides=(2,2,2), name='UpConv4')
    x = BNConv(x, filters=int(8/d), strides=(1,1,1), name='Conv12')
    x = tf.keras.layers.Conv3D(filters=1, kernel_size=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    
    CAE_Model = tf.keras.Model(inLayer, x)
    return CAE_Model
    
if __name__ == '__main__':
#    img_size = (None,None,None,1)
    img_size = (128,128,128,1)
    latent_dim = 64
    CAE3D = FullModel(img_size, latent_dim)
    CAE3D.summary()
    encoder = get_encoder_from_CAE3D(CAE3D)
#    decoder = get_decoder_from_CAE3D(CAE3D, latent_dim)
    
    tf.keras.utils.plot_model(CAE3D, to_file='./CAE_Model.png', show_shapes=True)
    tf.keras.utils.plot_model(encoder, to_file='./CAE_Encoder.png', show_shapes=True)
#    tf.keras.utils.plot_model(decoder, to_file='./CAE_Decoder.png', show_shapes=True)