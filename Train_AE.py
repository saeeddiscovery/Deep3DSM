# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 19:56:07 2018

@author: Saeed Mhq
"""

# In the name of GOD

import os, glob
from Utils.utils import myPrint, myLog
from Utils.utils import sortHuman

#model = 'CAE'
model = 'CVAE'
resultsDir = './Results/'

# Resume training
continueFromLatRun = False
lastRun = '/run-1/'

if not os.path.exists(resultsDir+model) or os.listdir(resultsDir+model)==[]:
    currRun = '/run-1/'
else:
    runs = glob.glob(os.path.join(resultsDir+model, 'run-*'))
    runs = sortHuman(runs)
    s = runs[-1]
    s = s.replace(" ", "")
    currID = [int(s) for s in s.split('-') if s.isdigit()][0]
    currRun = '/run-{}/'.format(currID+1)
    
currDir = resultsDir + model + currRun

if not os.path.exists(currDir):
    os.makedirs(currDir)
    
'''--------------Load Data--------------'''

from Utils.load_dataset import prepare_dataset, load_list

datasetDir = './Dataset/Dataset_Liver'

if continueFromLatRun:
   currDir = resultsDir + model + lastRun
   trainListPath = currDir + '/reports/training_list.txt'
   validListPath = currDir + '/reports/validation_list.txt' 
   dTrain, _ = load_list(trainListPath) 
   dValid, _ = load_list(validListPath) 
   myPrint('------------<  Dataset Info >------------', path=currDir)
   myPrint('...Dataset reloaded from saved lists', path=currDir)
   myPrint('...Train images:      {0}'.format(len(dTrain)), path=currDir)
   myPrint('...Validation images:      {0}'.format(len(dTrain)), path=currDir)
 
else:
    dTrain, dValid, _, _ = prepare_dataset(datasetDir, split=0.955, logPath=currDir)

## Add random noise before training!
#myPrint('...Adding noise to images N(0,0.33)', path=currDir)
#import numpy as np
#noise_factor = 0.5 
#dTrain_noisy = dTrain + noise_factor * np.random.normal(loc=0.0, scale=0.33, size=dTrain.shape)
#dValid_noisy = dValid + noise_factor * np.random.normal(loc=0.0, scale=0.33, size=dValid.shape) 

##-------Visualize Dataset-------#
#from Utils.utils import visualizeDataset
#visualizeDataset(dTrain[0:16], plotSize=[4,4])


'''--------------Build Model--------------'''

import tensorflow as tf
from Models import CAE_3D, CVAE_3D

import numpy as np
import datetime
K = tf.keras.backend

# GPU Memory Management
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
K.set_session(sess)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def myLoss(y_true, y_pred):
    a = 0.5
    BCE = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    DCE = -dice_coef(y_true, y_pred)
    myLoss = a*BCE + (1-a)*DCE
    return myLoss

def summary(model): # Compute number of params in a model (the actual number of floats)
    trainParams = sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
    myPrint('------------< Model Summary >------------', path=currDir)
    myPrint('...Total params:      {:,}'.format(model.count_params()), path=currDir)
    myPrint('...Trainable params:  {:,}'.format(trainParams), path=currDir)

img_size = dTrain.shape[1:]
#img_size = (None, None, None, 1)
latent_dim = 64
batch_size = 1
myPrint('...Input image size: {}'.format(img_size), path=currDir)
myPrint('...Batch size: {}'.format(batch_size), path=currDir)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

lr = 0.001
decay = 1e-3
#opt = tf.keras.optimizers.Adam(lr=0.01, decay=1e-5)
opt = tf.keras.optimizers.Adam(lr=lr)
lr_metric = get_lr_metric(opt)

if model == 'CAE':
    fullModel = CAE_3D.FullModel(img_size, latent_dim)
    encoder = CAE_3D.get_encoder_from_CAE3D(fullModel)
    fullModel.compile(optimizer=opt, loss=myLoss, metrics=[dice_coef, lr_metric],
            options=run_opts)

elif model == 'CVAE':
    encoder, generator, fullModel = CVAE_3D.CVAE(img_size, batch_size, latent_dim)
    tf.keras.utils.plot_model(generator, to_file=currDir+'/reports/' + model + '_3D_generator.png', show_shapes=True)
    fullModel.compile(optimizer=opt, loss=None, metrics=[lr_metric],
            options=run_opts)

summary(fullModel)
tf.keras.utils.plot_model(fullModel, to_file=currDir+'/reports/' + model + '_3D_Model.png', show_shapes=True)
tf.keras.utils.plot_model(encoder, to_file=currDir+'/reports/' + model + '_3D_encoder.png', show_shapes=True)
#    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

'''--------------Train Model--------------'''
myPrint('------------< Start Training >-----------', path=currDir)
start = datetime.datetime.now()
myPrint('...Start: {}'.format(start.ctime()[:-5]), path=currDir)

weightsDir = currDir+'/weights/'
if not os.path.exists(weightsDir):
    os.mkdir(weightsDir)

myLog('epoch\tlr\tloss\tval_loss', path=currDir)
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self,epoch, logs={}):
        # Things done on beginning of epoch. 
        return
    def on_epoch_end(self, epoch, logs={}):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
#        iterations = self.model.optimizer.iterations
#        lr_with_decay = lr / (1. + decay * (epoch)//2))
        lr_with_decay = lr / (1. + decay * epoch)
        myLog(str(epoch) +'\t' + str(K.eval(lr_with_decay)) +'\t' + str(logs.get("loss")) +'\t' + str(logs.get("val_loss")), path=currDir)

weights_file_v = weightsDir + model + "_3D_model_v.hdf5"
weights_file_t = weightsDir + model + "_3D_model_t.hdf5"

#if continueFromLatRun:
#    fullModel.load_weights(weights_file)

epochs=200        
#weights_file = "CAE_3D_model-{epoch:02d}-{val_loss:.2f}.hdf5"
model_checkpoint_v = tf.keras.callbacks.ModelCheckpoint(weights_file_v,
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True)
model_checkpoint_t = tf.keras.callbacks.ModelCheckpoint(weights_file_t,
                                                      monitor='loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True)
logger = tf.keras.callbacks.CSVLogger(currDir+'/reports/training.log', separator='\t')
tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/'+model+currRun)
lrs = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr / (1. + decay * epoch))
callbacks = [tensorBoard, model_checkpoint_v, model_checkpoint_t, logger, MyCallback(), lrs]
if model == 'CAE':
    fullModel.fit(dTrain_noisy, dTrain, shuffle=True, epochs=epochs, batch_size=batch_size,
           validation_data=(dValid_noisy, dValid), callbacks=callbacks)
if model == 'CVAE':
    fullModel.fit(dTrain, shuffle=True, epochs=epochs, batch_size=batch_size,
               validation_data=(dValid, None), callbacks=callbacks)
    generator.save_weights(weightsDir+"/" + model + "_3D_generator.hdf5")
    
encoder.save_weights(weightsDir+"/" + model + "_3D_encoder.hdf5")

end = datetime.datetime.now()
elapsed = end-start
myPrint('...End: {}'.format(end.ctime()[:-5]), path=currDir)
myPrint('...Train time: {}'.format(elapsed), path=currDir)

'''---------Visualize latent space---------'''
#TODO

'''-----------Generate new images----------'''
#TODO
