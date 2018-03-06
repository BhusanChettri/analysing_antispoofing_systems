"""
Functions for explaining classifiers that use Image data.
"""
import copy

import numpy as np
import sklearn
import sklearn.preprocessing
import librosa.display as disp
import matplotlib.pyplot as plt

from . import lime_base


class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.distance = {} # SLIME

    def get_image_and_mask(self, label, positive_only=True, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that contribute to
                the prediction of the label. Otherwise, use the top
                num_features superpixels, which can be positive or negative
                towards the label
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            print('Label not in explanation')
            return
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        distance = self.distance[label] #SLIME
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            #SLIME
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]

            for x in exp:
                print("Component index: %d weight %f" %(x[0], x[1]))
            '''fs = [x[0] for x in exp
                  if x[1] < 0][:num_features]'''
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask,(fs, distance)        # SLIME : added fs here
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = 1 if w < 0 else 2
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
                for cp in [0, 1, 2]:
                    if c == cp:
                        continue
                    # temp[segments == f, cp] *= 0.5
            return temp, mask


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, verbose=False,
                 feature_selection='auto'):
        """Init function.

        Args:
            training_data: numpy 2d array
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True. Options
                are 'quartile', 'decile' or 'entropy'
        """
        kernel_width = float(kernel_width)

        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel, verbose)

    def explain_instance(self, image, classifier_fn, 
                         sess, modelScore, input_data, keep_prob1, keep_prob2, keep_prob3,
                         labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         distance_metric='cosine', model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array, corresponding to a row
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        from skimage.segmentation import quickshift
        '''segments = quickshift(image, kernel_size=4,
                              max_dist=200, ratio=0.2)'''
        # SLIME
        # Lame way of segmenting the input.
        # Change to performant way : Todo
        '''from skimage.segmentation import slic
        segments = slic(image, n_segments= 24, compactness=1)'''
        segments = np.empty((image.shape[0], image.shape[1]))
        '''for i in range(0, image.shape[0]):
            if i <19:
                segments[i][0:20]= 0
                segments[i][20:40]= 1
                segments[i][40:60]= 2
                segments[i][60:80]= 3
            elif i>=19 and i<38:
                segments[i][0:20] = 4
                segments[i][20:40] = 5
                segments[i][40:60] = 6
                segments[i][60:80] = 7            
            elif i>=38 and i<57:
                segments[i][0:20] = 8
                segments[i][20:40] = 9
                segments[i][40:60] = 10
                segments[i][60:80] = 11
                
            elif i>=57 and i<76:
                segments[i][0:20] = 12
                segments[i][20:40] = 13
                segments[i][40:60] = 14
                segments[i][60:80] = 15
                
            elif i>=76 and i<95:
                segments[i][0:20] = 16
                segments[i][20:40] = 17
                segments[i][40:60] = 18
                segments[i][60:80] = 19
                
            else:
                segments[i][0:20] = 20
                segments[i][20:40] = 21
                segments[i][40:60] = 22
                segments[i][60:80] = 23'''
        
        # Temporal segmentation: code is hard-coded for 3sec audio to be segmented into 10 super-samples.
        for i in range(0, image.shape[0]):
            if i <30:
                segments[i]= 0
            elif i>=30 and i<60:
                segments[i]= 1
           
            elif i>=60 and i<90:
                segments[i] = 2
                
            elif i>=90 and i<120:
                segments[i]= 3
            
            elif i>=120 and i<150:
                segments[i]= 4
           
            elif i>=150 and i<180:
                segments[i] = 5
                
            elif i>=180 and i<210:
                segments[i]= 6
            
            elif i>=210 and i<240:
                segments[i]= 7
           
            elif i>=240 and i<270:
                segments[i] = 8
                    
            else:
                segments[i]= 9 
                
        '''plt.figure(1)
        disp.specshow(segments.T)
        plt.savefig('/Users/Saumitra/Documents/workspace/analysing_antispoofing_systems/temp.png')'''       
        
        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        sess, modelScore, input_data, keep_prob1, keep_prob2, keep_prob3,
                                        batch_size=batch_size)

        #SLIME
        print("Labels assigned via LIME path: %f %f" %(labels[0][0], labels[0][1]))

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
            
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score,         # SLIME added distance
             ret_exp.distance[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp, segments    # SLIME added segments return

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    sess, modelScore, input_data, keep_prob1, keep_prob2, keep_prob3,
                    batch_size=10):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = np.random.randint(0, 2, num_samples * n_features).reshape(
                (num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(np.expand_dims(imgs, axis=3)), sess, modelScore, input_data, keep_prob1, keep_prob2, keep_prob3)
                #preds = classifier_fn(np.array(imgs))
                labels.extend(preds[0]) # taking only first element
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(np.expand_dims(imgs, axis=3)), sess, modelScore, input_data, keep_prob1, keep_prob2, keep_prob3)
            #preds = classifier_fn(np.array(imgs))
            labels.extend(preds[0]) # taking only first element
        return data, np.array(labels)
