# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 16:04:07 2016

@author: Saeed Mhq
"""
# In the name of GOD

import os, glob
from random import shuffle
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
if __name__ == '__main__':
    from utils import myPrint
else:
    from .utils import myPrint

def split_list(input_list, split, shuffleList=True):
    if shuffleList:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training_list = input_list[:n_training]
    validation_list = input_list[n_training:]
    return training_list, validation_list

def load_images(images_list, padSize, ScaleFactor):
    noImages = int(images_list.__len__())
    
    if (noImages == 0):
        return []
    
    dImages = list()
    dImages_zoomed = list()

    for i,file in enumerate(images_list):
        imgFilename = os.path.join(file)
        imgInput = sitk.ReadImage(imgFilename)
        img = sitk.GetArrayFromImage(imgInput)
        img = np.asarray(img, dtype='float32')
#        img = np.rot90(img)
#        img = np.fliplr(img)
        img = np.flipud(img)
        if padSize > 0:
            dImages.append(np.pad(img, padSize, 'edge'))
        else:
            dImages.append(img)
        
#    dTrain_pad = np.zeros(shape=[noTrain,img.shape[0]+padSize*2,img.shape[1]+padSize*2,img.shape[2]+padSize*2], dtype='float32')
#    dTrain_zoomed = np.zeros(shape=[noTrain,int(dTrain_pad.shape[1]*ScaleFactor),int(dTrain_pad.shape[2]*ScaleFactor),int(dTrain_pad.shape[3]*ScaleFactor)], dtype='float32')
#    for i in range(len(dTrain)):
#        dTrain_pad[i,:,:,:] = np.pad(dTrain[i], padSize, 'constant')

    dImages = rescale(dImages)

    if ScaleFactor == None:
        dImages = np.asarray(dImages)
        dImages = np.array(dImages[:,:,:,:,np.newaxis], dtype='float32')
    else:
        for i in range(noImages):
            dImages_zoomed.append(ndimage.zoom(dImages[i], ScaleFactor, order=0))
#            dTrain_zoomed[i,:,:,:] = ndimage.zoom(dTrain[i], ScaleFactor, order=0) #Pooling
        dImages = np.asarray(dImages_zoomed)
        dImages = dImages[:,:,:,:,np.newaxis]
        
    return dImages

def save_list(images_list, fileName):
    # save train/validation files in a text file
    if not os.path.exists(os.path.dirname(fileName)):
        os.mkdir(os.path.dirname(fileName))
    i=0
    with open(fileName, 'w') as imagesListFile:
        for item in images_list:
            imagesListFile.write(str(i)+"\t{}\n".format(item))
            i+=1
    
def prepare_dataset(datasetDir, split=0.8, padSize=0, shuffle=True, scaleFactor=None, logPath='.'):
    """ 
    Function that loads 3D medical image data
    and prepare it for training
    
    Arguments:
        - datasetDir: The directory that contains all dataset images
        - split: The ratio (0-1) of images for the training data
        - padSize: The number of voxels to pad
            `Default`: zero (no padding)
        - shuffle_list: Defines if the list of files should be shuffled or not
                `Default`: True (shuffle the images list)
        - ScaleFactor: Defines the scale of the data (0.5 -> 1/2 size)
            `Default`: None (No scale)
    """

#    img_paths = r'./Dataset/*.nii.gz'
    img_paths = os.path.join(datasetDir, '*.nii.gz')
    img_addrs = glob.glob(img_paths)

# Second method for Trian/Valid images list creation
    training_list, validation_list = split_list(img_addrs, split=split, shuffleList=shuffle)

    dTrain = load_images(training_list, padSize, scaleFactor)
    save_list(training_list, logPath+'/reports/training_list.txt')
    myPrint('------------<  Dataset Info >------------', path=logPath)
    myPrint('...Train images:      {0}'.format(len(dTrain)), path=logPath)
    dValid = load_images(validation_list, padSize, scaleFactor)
    save_list(validation_list, logPath+'/reports/validation_list.txt')
    myPrint('...Validation images: {0}'.format(len(dValid)), path=logPath)

    return dTrain, dValid, training_list, validation_list

	
def load_list(imagesListFile, scaleFactor=None):
    with open(imagesListFile, 'r') as fileList:
        files = fileList.readlines()
    imgFiles = []
    fileNames = []
    for i in range(len(files)):
        imgFiles.append(files[i].split('\t')[1][:-1])
        fileNames.append(os.path.basename(files[i].split('\t')[1])[:-1])
    dImages = load_images(imgFiles, padSize=0, ScaleFactor=scaleFactor)
#    print('------------<  Dataset Info >------------')
#    print('...Test images:      {0}'.format(len(dImages)))
    return dImages, fileNames

def normalize(data):
    count = len(data)
    for i in range(count) :
        data[i] = np.add(data[i], -np.mean(data[i][:]))
        data[i] /= np.std(data[i][:])
#        data[i] = np.add(data[i], -np.min(data[i]))
#        data[i] /= np.add(np.max(data[i]), -np.min(data[i]))
        #data[i,:] = np.add(data[i,:], -np.min(data[i,:]))
    return data

def rescale(data):
    for i, data_rescaled in enumerate(data):
        data[i] /= np.max(data_rescaled[:])
#        data[i] -= np.mean(data_rescaled[:])
#        data[i] /= np.std(data_rescaled[:])
#        data[i] -= 0.5
#        data[i] *= 2.
        data[i] = np.asarray(data[i], 'float32')
    return data

if __name__ == '__main__':
    datasetDir = './Dataset_Liver'
    padSize = 0
    dTrain, dValid, training_list, validation_list = prepare_dataset(datasetDir)