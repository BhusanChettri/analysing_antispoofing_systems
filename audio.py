'''
Created on 6 Dec 2017

@author: Saumitra
Reads the saved input representations, normalises
it and prepares the data to be fed to the network
for prediction.
'''

import numpy as np

def read_audio(pow_spects_file, mean_std_file):
    try:
        with np.load(pow_spects_file) as input_file:
            pow_spects = input_file['spectrograms']
    except (IOError, KeyError):
        print("Input spectrogram file is not present or fails to open!!")
    print(pow_spects[0].shape)   
    try:
        with np.load(mean_std_file) as mean_std_file:
            mean = mean_std_file['mean']
            std = mean_std_file['std']
    except (IOError, KeyError):
        print("Mean/Std file is not present or fails to open!!")
        
    pow_spects_norm = [(spects - mean) * np.reciprocal(std) for spects in pow_spects]
    
    return pow_spects_norm

  
