# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 19:56:07 2018

@author: Saeed Mhq
"""

# In the name of GOD

import os, glob

model = 'CAE'
#model = 'CVAE'

import re
def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9])', key)]
    l.sort(key=alphanum)
    return l

resultsDir = './Results/'
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)
if os.listdir(resultsDir)==[]:
    currRun = '/run1'
    os.mkdir(resultsDir+currRun)
else:
    runs = glob.glob(os.path.join(resultsDir, 'run*'))
    runs = sort_human(runs)
    currRun = '/run' + str(int(runs[-1][13:])+1)
    os.mkdir(resultsDir+currRun)
    
'''--------------Load Data--------------'''

from Utils.load_dataset import prepare_dataset
 
datasetDir = './Dataset/Dataset_Liver'
dTrain, dValid, _, _ = prepare_dataset(datasetDir, logPath=resultsDir+currRun)

#-------Visualize Dataset-------#
#from Utils.utils import visualizeDataset
#visualizeDataset(dTrain, plotSize=[4,4])

'''--------------Build Model--------------'''

import tensorflow as tf
from Models import CAE_3D, CVAE_3D
from Utils.utils import myPrint
import numpy as np
import datetime
K = tf.keras.backend
    
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def summary(model): # Compute number of params in a model (the actual number of floats)
    trainParams = sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
    myPrint('------------< Model Summary >------------', path=resultsDir+currRun)
    myPrint('...Total params:      {:,}'.format(model.count_params()), path=resultsDir+currRun)
    myPrint('...Trainable params:  {:,}'.format(trainParams), path=resultsDir+currRun)

img_size = dTrain.shape[1:]
latent_dim = 128
batch_size = 1

if model == 'CAE':
    encoder = CAE_3D.Encoder(img_size, latent_dim)
    decoder = CAE_3D.Decoder(latent_dim)
    inLayer = tf.keras.layers.Input(shape=img_size)
    CAE_3D = tf.keras.Model(inLayer,decoder(encoder(inLayer)))
    CAE_3D.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=dice_coef_loss, metrics=['accuracy'])
    summary(CAE_3D)
    tf.keras.utils.plot_model(CAE_3D, to_file=resultsDir+currRun+'/reports/CAE_3D.png', show_shapes=True)
    tf.keras.utils.plot_model(encoder, to_file=resultsDir+currRun+'/reports/CAE_Encoder.png', show_shapes=True)
    tf.keras.utils.plot_model(decoder, to_file=resultsDir+currRun+'/reports/CAE_Decoder.png', show_shapes=True)

elif model == 'CVAE':
    encoder, generator, CVAE_3D = CVAE_3D.CVAE(img_size, batch_size, latent_dim)
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    CVAE_3D.compile(optimizer=opt, loss=None)
    summary(summary(CVAE_3D))
    tf.keras.utils.plot_model(CVAE_3D, to_file=resultsDir+currRun+'/reports/CVAE_3D.png', show_shapes=True)
    tf.keras.utils.plot_model(encoder, to_file=resultsDir+currRun+'/reports/CVAE_encoder.png', show_shapes=True)
    tf.keras.utils.plot_model(generator, to_file=resultsDir+currRun+'/reports/CVAE_generator.png', show_shapes=True)

'''--------------Train Model--------------'''
myPrint('------------< Start Training >-----------', path=resultsDir+currRun)
start = datetime.datetime.now()
myPrint('Start: {}'.format(start.ctime()[:-5]), path=resultsDir+currRun)
epochs = 100

weightsDir = resultsDir+currRun+'/weights'
if not os.path.exists(weightsDir):
    os.mkdir(weightsDir)
    
if model == 'CAE':
    #model_file = "CAE_3D_model-{epoch:02d}-{val_loss:.2f}.hdf5"
    model_file = weightsDir+"/CAE_3D_model.hdf5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file,
                                                          monitor='loss',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          save_weights_only=True)
    logger = tf.keras.callbacks.CSVLogger(resultsDir+currRun+'/reports/training.log', separator='\t')
    tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/CAE'+currRun)
    callbacks = [tensorBoard, model_checkpoint, logger]
    CAE_3D.fit(dTrain, dTrain, shuffle=True, epochs=epochs, batch_size=batch_size,
               validation_data=(dValid, dValid), callbacks=callbacks)

elif model == 'CVAE':
    #model_file = "CAE_3D_model-{epoch:02d}-{val_loss:.2f}.hdf5"
    model_file = weightsDir+"/CAE_3D_model.hdf5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file,
                                                          monitor='loss',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          save_weights_only=True)
    logger = tf.keras.callbacks.CSVLogger(resultsDir+currRun+'/reports/training.log', separator='\t')
    tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/CVAE'+currRun)
    callbacks = [tensorBoard, model_checkpoint, logger]
    CVAE_3D.fit(dTrain, shuffle=True, epochs=epochs, batch_size=batch_size,
               validation_data=(dValid, None), callbacks=callbacks)

end = datetime.datetime.now()
elapsed = end-start
myPrint('Start: {}'.format(start.ctime()[:-5]), path=resultsDir+currRun)
myPrint('Train time: {}'.format(elapsed), path=resultsDir+currRun)

'''---------Visualize latent space---------'''
#TODO

'''-----------Generate new images----------'''
#TODO
