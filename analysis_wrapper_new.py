'''
Created on 6 Dec 2017

@author: Saumitra
wrapper file to call other modules.
'''
import audio
import matplotlib
matplotlib.use('Agg')

import matplotlib as mpl
#mpl.use('TkAgg')     # needed as there is an issue with default matplotlib backend on macosx (https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python)


import matplotlib.pyplot as plt
import librosa.display as disp
import tensorflow as tf
import os
import nn_architecture
import numpy as np
from lime import lime_image
#from final_code import nn_architecture
import saliency

import time

def write_scores_to_file(prediction, after_decimal=5, outfile='prediction.txt'):
    print('Prediction list length = ', len(prediction))
    posteriors = [vector for scoreList in prediction for scores in scoreList for vector in scores]
    with open(outfile, 'w') as f:
        for probs in posteriors:
            gen=probs[0]
            spoof=probs[1]
                         
            score = np.log(gen) - np.log(spoof)
            f.write("%.4f\n" % (score))

def normalise(x):
    """
    Normalise a vector/ matrix, in range 0 - 1
    """
    return((x-x.min())/(x.max()-x.min()))

def reshape_minibatch(minibatch_data): #, minibatch_labels):
    # inputs is a list of numpy 2d arrays.
    # Ouput: 4d tensor of minibatch data and 2d label arrays
    
    l = len(minibatch_data)
    print(l)
    t, f = minibatch_data[0].shape
    print('Time and frequency', t, f)
    
    reshaped_data = np.empty((l,t,f))
    for i in range(l):
        reshaped_data[i] = minibatch_data[i]
    
    print('New 3d shape = ', reshaped_data.shape)
    
    # Now convert 3d array to 4d array
    reshaped_data = np.expand_dims(reshaped_data, axis=3)
    print('New 4d shape = ', reshaped_data.shape)    
    
    # Re-arrange binary labels in one-hot 2 dimensional vector form
    #new_labels = [[1,0] if label == 1 else [0,1] for label in minibatch_labels]
    
    return np.asarray(reshaped_data) #, np.asarray(new_labels)

#------------------------------------------------------------------------------------------------------------------------------    
def iterate_minibatches(inputs, batchsize, shuffle=False):
    '''
    Generator function for iterating over the dataset.
    
    inputs = list of numpy arrays
    targets = list of numpy arrays that hold labels : TODO , need to fix the labels properly    
    '''
        
    #assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))    
    np.random.shuffle(indices)    

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):  #total batches                 
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:            
            excerpt = slice(start_idx, start_idx + batchsize)
 
        yield np.asarray(inputs)[excerpt]#, np.asarray(targets)[excerpt]
    
#------------------------------------------------------------------------------------------------------------------------------
def makeDirectory(path):
    # Create directories for storing tensorflow events    
    if not os.path.exists(path):
        os.makedirs(path)

def computeModelScore(model_prediction, apply_softmax=True):
    # model_prediction is the output from the output layer which is in logits.
    # we apply softmax to get probability depending upon flag variable being passed
    
    if apply_softmax:
        prediction = tf.nn.softmax(model_prediction)
    else:
        prediction = model_prediction
        
    return prediction   #writeOutput(prediction)

def load_model(save_path, n_model=None):
    
    print('Loading model parameters ...')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    
    if n_model==None:        
        #path = tf.train.latest_checkpoint(save_path)
        path = os.path.join(save_path,"bestModel.ckpt")
    else:        
        path = os.path.join(save_path,"model.ckpt-"+str(n_model))
    #print(path)        
    saver.restore(sess, path)
    
    return sess, saver


def prediction_fn(data, sess, model_score, input_data, keep_prob1, keep_prob2, keep_prob3):
    scores = sess.run([model_score], feed_dict={input_data:data, keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})  
    return scores


def main():
    
    
    #1. Top Genuine correct file index = 575   (Gen/spoofed prob + score = 0.999954/4.55669e-05  9.99628)
    #2. Top Genuine in-correct file index = 286  (0.0622482/0.937752, -2.71236)    
    #3. Top Spoofed correct file index = 1535   (4.90329e-05 /0.999951, -9.92297)
    #4. Top Spoofed in-correct file index = 906  (0.996418/ 0.00358186, 5.62828)
    
    
    # Top Genuine Correct   
    file_idx = 574  # 575-1        # Id of the file for which we want predictions    
    savePath = 'explanations/topGenuine_correct/'
    
    # Top Genuine Incorrect    
    #file_idx = 285  #(286-1)  
    #savePath = 'explanations/topGenuine_Incorrect/'
        
    # Top Spoofed Correct   
    #file_idx = 1534  # 1535-1
    #savePath = 'explanations/topSpoofed_correct/'
    
    # Top Spoofed Incorrect    
    #file_idx = 905  #(906-1)  
    #savePath = 'explanations/topSpoofed_Incorrect/'

    makeDirectory(savePath)
    runs = 5   # how many runs you want to run the same setup, to ensure predictions are correct
        
    sampling_rate = 16000
    hop_size = 160
    
    pow_spects_file = 'dev_spec.npz'
    mean_std_file = 'mean_std_trainData.npz'
    
    model_path = 'model_3sec_relu_0.5_run9' 
    dataType = 'test'
    trainSize = 3
    init_type = 'xavier'
    fftSize = 256
    f = 129
    t=300
    padding = 'SAME'
    targets= 2
    act = tf.nn.relu    

    plot = False
    debug_prints = True
    
    # Returns a list where each element is a normalized power spectrogram per file.
    print('Loading spectrograms...')
    norm_pow_spects = audio.read_audio(pow_spects_file, mean_std_file)
    data = norm_pow_spects
    print(data[0].shape)
    
    if debug_prints:
        print('Input spectrogram shape: (%d, %d)' %(norm_pow_spects[file_idx].shape))
    
    if plot:
        # figure 1
        print('Plotting spectrogram for file number %d' %(file_idx + 1))
        plt.figure(1)
        plt.subplot(1, 1, 1)
        disp.specshow(norm_pow_spects[file_idx].T, sr = sampling_rate, y_axis = 'linear', x_axis = 'time', hop_length = hop_size, cmap = 'coolwarm')
        plt.title('Input spectrogram')
        plt.colorbar()
        plt.show()
     
        
    print('Reset TF graph and load trained models and session..')        
    #tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        input_data = tf.placeholder(tf.float32, [None, t, f,1])
                
        # Placeholders for droput probability
        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)
        keep_prob3 = tf.placeholder(tf.float32) 
    
        # Get model architecture that was used during training            
        featureExtractor, model_prediction, network_weights, activations, biases= nn_architecture.cnnModel3(dataType, trainSize, input_data, act, init_type, targets, fftSize, padding, keep_prob1, keep_prob2, keep_prob3)        
        modelScore = computeModelScore(model_prediction, apply_softmax=True)    
        
        #Load trained session and model parameters    
        sess, saver = load_model(model_path)
        print('Model parameters loaded succesfully !!')
        
        # saliency code
        
        neuron_selector = tf.placeholder(tf.int32)
        y = model_prediction[0][neuron_selector]
    
    
    for run in range(0,runs):
        
        num_feat = 2 # number of components per explanation , e.g., select top 1 component out of the given 10.
        n_samples = 10 # number of synthetic samples generated by SLIME
        top_lab =  2 # tells in a multi-class scenario how many labels to be explained: we can keep it fixed
        label_idx = 0 # tells what class needs to be explained, 0 - genuine, 1 - spoofed
        exp_type = 2 # 0: time segmentation, 1: frequency segmentation, 2: time-frequency segmentation
        
        print('Generating explanations for Run: ', run+1)
        explainer = lime_image.LimeImageExplainer(verbose=True)
        t1 = time.time()
        # generation explanation
        explanation, seg = explainer.explain_instance(data[file_idx].reshape(t,f), prediction_fn , sess, modelScore, input_data, keep_prob1, keep_prob2, keep_prob3, hide_color=0, top_labels=top_lab, num_samples=n_samples, seg_type = exp_type)
        print ("time taken for explanation generation:%f" %(time.time() - t1))
        
        # extracting the information from the generated explanation
        # temp : masked spectrogram
        # mask : binary mask, 1: presence of the component, 0: absence of the component
        # fs : a tuple of two objects, one is a list of enabled components, other is the prediction error
        # I am also printing the weights assigned to each component. It will be very useful in analysis, something I didn't do in SLIME paper.
        # At the moment, I am only returning component index that positively influences a prediction.
        temp, mask, fs = explanation.get_image_and_mask(label_idx, positive_only=True, hide_rest=True, num_features=num_feat)
               
        print(fs)

        # plotting the results    
        plt.figure(2)
        plt.subplot(2,1,1)
        disp.specshow(normalise((data[file_idx].reshape(t, f))).T, y_axis= 'linear', x_axis='off', sr=sampling_rate, hop_length=hop_size, cmap = 'coolwarm')
        plt.title('Input Spectrogram')
        plt.subplot(2,1,2)
        disp.specshow(normalise(temp).T, y_axis= 'linear', x_axis='time', sr=sampling_rate, hop_length=hop_size, cmap = 'coolwarm')
        plt.title('Top %d explanations generated by LIME' %num_feat)
    
    
        filename = savePath + 'file_'+str(file_idx+1) + '_run' + str(run+1) + '.png' # during final plots change the file extension type to .pdf and keep dpi = 300
    
        plt.savefig(filename)
        plt.close()


    # saliency maps experiments
    # may be some issue with graph
    
    '''print(prediction_fn(np.expand_dims(np.expand_dims(data[file_idx], axis=0), axis=3), sess, modelScore, input_data, keep_prob1, keep_prob2, keep_prob3))
    
    gradient_saliency = saliency.GradientSaliency(graph, sess, y, input_data)
    
    # Compute the vanilla mask and the smoothed mask.
    vanilla_mask_3d = gradient_saliency.GetMask(np.expand_dims(data[file_idx], axis=3), feed_dict = {keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, neuron_selector: label_idx})
    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(np.expand_dims(data[file_idx], axis=3), feed_dict = {keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, neuron_selector: label_idx})
    
    print(vanilla_mask_3d.shape)
    
    # plotting the results    
    plt.figure(3)
    plt.subplot(3,1,1)
    disp.specshow(normalise(data[file_idx]).T, y_axis= 'linear', x_axis='off', sr=sampling_rate, hop_length=hop_size, cmap = 'coolwarm')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    disp.specshow((vanilla_mask_3d.reshape(t, f)).T, y_axis= 'linear', x_axis='off', sr=sampling_rate, hop_length=hop_size, cmap = 'coolwarm')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    disp.specshow(normalise(smoothgrad_mask_3d.reshape(t, f)).T, y_axis= 'linear', x_axis='off', sr=sampling_rate, hop_length=hop_size, cmap = 'coolwarm')
    plt.colorbar()
    plt.savefig('saliency.png')'''
 


if __name__ == '__main__':
    main()
