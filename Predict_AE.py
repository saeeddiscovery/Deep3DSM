# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 12:45:25 2018

@author: Saeed Mhq
"""

# In the name of GOD

import tensorflow as tf
K = tf.keras.backend
num_cores = 4
GPU = True
if GPU:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 1
    num_GPU = 0
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


from Models import CAE_3D, CVAE_3D
from Utils.load_dataset import load_list

#modelName = 'CAE'
modelName = 'CVAE'
noisy = False
resultsDir = './Results/'
run = 'run-2'
currDir = resultsDir + modelName  + '/' + run

listPath = currDir + '/reports/validation_list.txt' 
testImages, testFiles = load_list(listPath)

# Add random noise before training!
if noisy:
    print('...Adding noise to images N(0,0.33)')
    import numpy as np
    noise_factor = 0.5 
    testImages = testImages + noise_factor * np.random.normal(loc=0.0, scale=0.33, size=testImages.shape)

img_size = testImages.shape[1:]
latent_dim = 128
if modelName == 'CAE':
    model = CAE_3D.FullModel(img_size, latent_dim)
    encoder = CAE_3D.get_encoder_from_CAE3D(model)

elif modelName == 'CVAE':
    batch_size = 1
    encoder, generator, model = CVAE_3D.CVAE(img_size, batch_size, latent_dim)
    
weightsPath =currDir + '/weights/' + modelName + '_3D_model.hdf5'
model.load_weights(weightsPath)
    
    
'''--------------Prediction---------------'''
import numpy as np
import os
import SimpleITK as sitk

print('------------<  Dataset Info >------------')
print('Model: ' + modelName)
predDir = currDir + '/Predicted/'
if not os.path.exists(predDir):
    os.mkdir(predDir)
predicted = []
threshold = 0.7
for i,img in enumerate(testImages):
    img = img[np.newaxis,:]
    print('... Predicting ' + testFiles[i])
    predicted = np.squeeze(model.predict(img))
    predicted = predicted[::-1]
    predicted[predicted <= threshold] = 0
    predicted[predicted > threshold] = 1
    volOut = sitk.GetImageFromArray(predicted)
    outFile = os.path.join(predDir, testFiles[i][:-7]+'_pred.nii.gz')
    sitk.WriteImage(volOut, outFile)
    print('saved as ' + outFile)
