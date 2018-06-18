# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 12:45:25 2018

@author: Saeed Mhq
"""

# In the name of GOD

from Models import CAE_3D
from Utils.load_dataset import load_list

modelName = 'CAE'

run = 'run1'
resultsPath = './Results/' + run
listPath = resultsPath + '/reports/validation_list.txt' 
testImages, testFiles = load_list(listPath)

# Add random noise before training!
print('...Adding noise to images N(0,0.33)')
import numpy as np
noise_factor = 0.5 
dTrain_noisy = testImages + noise_factor * np.random.normal(loc=0.0, scale=0.33, size=testImages.shape)

img_size = testImages.shape[1:]
latent_dim = 128
if modelName == 'CAE':
    model = CAE_3D.FullModel(img_size, latent_dim)
    weightsPath =resultsPath + '/weights/' + modelName + '_3D_model.hdf5'
    model.load_weights(weightsPath)
    encoder = CAE_3D.get_encoder_from_CAE3D(model)
    
'''--------------Prediction---------------'''
import numpy as np
import os
import SimpleITK as sitk

print('------------<  Dataset Info >------------')
print('Model: ' + modelName)
predDir = resultsPath + '/Predicted/'
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
