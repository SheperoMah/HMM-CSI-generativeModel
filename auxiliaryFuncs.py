#!/usr/bin/env python3

import numpy as np
from hmmlearn import hmm
from copy import deepcopy
from hmmlearn import utils
import scipy.stats as stts
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
	""" Changes the transition matrix of the HMM model. It scales the off-diagonal elements of the randomMat, and renormalize it, then returns a new model with the scaled transition matrix. c has to be in range (0,1].
	"""
	assert c > 0 and c <= 1, "The factor c has to be in the range (0,1]" 
	newMatrix = change_off_diagonal_prob(randomMat, c)      
	modelNew = deepcopy(model)
	modelNew.transmat_ = newMatrix
	return(modelNew) 

def train_hmm_model(trainingData, lengthSequences, numHiddenStates, dimObservations, numIterations, tolerance, randomSeed):
	""" Trains a Gaussian Mixture HMM model. It initializes the transmission matrix, the means and covariances
	"""
	np.random.seed(randomSeed)
	model = hmm.GaussianHMM(n_components=numHiddenStates, init_params="stmc", n_iter=numIterations, tol=tolerance)
	model.n_features = dimObservations
	model.fit(trainingData, lengthSequences)
	return(model)

def divide_data(data):
	""" Divides data into training, validation, and test data sets. The division ratio depends on the variable ratio which should be of length 2. Training data are placed in rows.
	"""
	numDataPoints = data.shape[0]
	indexes = np.arange(numDataPoints)
	idxTrain, idxVal = indexes[0:numDataPoints:1], indexes[1:numDataPoints:1]	
	training, validation = data[idxTrain, :], data[idxVal, :] 
	return(training, validation)

def calculate_length_sequence(mtrx):
	numberOfFeatures = mtrx.shape[1]
	numberOfObs = mtrx.shape[0]
	return([numberOfFeatures for i in range(numberOfObs)])

def score_model_on_datasets(model, dataSets, lengths):
	scores = [model.score(i[0], i[1]) for i in list(zip(dataSets, lengths))]
	return(scores)

def ks_test(model, dataSets):
	modelSamples = [model.sample(len(v)) for i, v in enumerate(dataSets)]
	scores = [stts.ks_2samp(i[0][0].flatten(), i[1].flatten()).statistic for i in list(zip(modelSamples, dataSets))]
	return(scores)

