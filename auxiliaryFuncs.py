#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
from hmmlearn import hmm
from hmmlearn import utils
from sklearn.externals import joblib

def change_off_diagonal_prob(T, constant):
    """ This function scales the off-diagonal of a transition matrix. Each r
ow should sum to 1.0 in the imput and output matrices.
    """     
	size = T.shape[0]
	offDiag =  T - T * np.eye(size)
	scaledOffDiag = constant * offDiag 
	scaledT = scaledOffDiag + np.diag(1 - scaledOffDiag.sum(axis=1))
	return(scaledT)

def change_transition_matrix_of_model(model, randomMat, c):
	""" Changes the transition matrix of the HMM model. It scales the off-diagonal elements of the randomMat, and renormalize it, then returns a new model with the scaled transition matrix.
	c has to be larger in range (0,1].
	"""
	assert c > 0 and c <= 1, "The factor c has to be in the range (0,1]" 
	newMatrix = change_off_diagonal_prob(randomMat, c)      
    modelNew = deepcopy(model)
    modelNew.transmat_ = newMatrix
    return(modelNew) 

def train_hmm_model(trainingData, lengthSequences, numHiddenStates, dimObservations, numIterations, randomSeed)
	""" Trains a Gaussian Mixture HMM model. It initializes the transmission matrix, the means and covariances
	"""
	np.random.seed(randomSeed)
	model = hmm.GaussianHMM(n_components=numHiddenStates, init_params="stmc", n_iter=numIterations)
	model.n_features = dimObservations
	model.fit(trainingData, lengthSequences)
	return(model)

	


