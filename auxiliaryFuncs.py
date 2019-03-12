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


def clean_axes(ax):
	""" Removes the right and top axes of matplotlib
	"""
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

def eclidean_distance_in_cluster(X, mean):
	""" Calculates the Eclidean distance of the cluster from a matrix and the mean of vector.
		D = sum_{i=1}^{n_k} sum_{j=1}^{n_k} ||x_i - x_j||^2
		  = 2 * n_k * sum_{i=1}^{n_k} ||x_i - \mu_k||^2.
		  See: https://datasciencelab.wordpress.com/tag/gap-statistic/

	Parameters
	----------
	X : np.array
		Matrix of observations, with shape n*p where p is the length of the
		vector, i.e., the number of features, and n are the number of observations.
	mean : np.array
		The mean of the cluster. A vector with length p.

	Returns
	-------
	float :
		Distance.
	"""
	n, p = X.shape
	assert p == mean.shape[0], "The length of the mean should be equal to the \
	number of columns in the X."

	distance = X - mean
	eclideanDistance = np.linalg.norm(X-mean, axis=1) ** 2
	D = 2 * n * np.sum(eclideanDistance)
	return(D)

def estimate_wk(drs, nrs):
	""" W_k = \sum_{i=1}^{K} 1/(2*n_i) * D_i
	"""
	wk = np.sum(drs / (2 * nrs))
	return(wk)

def gap_statistic(wksStar, wk):
	return(np.mean(wksStar) - wk)

def estimate_wk_model(model, data):
	k = model.means_.shape[0]
	_, clusters = model.decode(data)
	drs = np.zeros(shape=k)
	nrs = np.zeros(shape=k)

	for i in range(k):
		idx = clusters == i
		dataInCluster = data[idx]
		meanCluster = model.means_[i]
		drs[i] = eclidean_distance_in_cluster(dataInCluster, meanCluster)
		nrs[i] = dataInCluster.shape[0]

	wk = estimate_wk(drs, nrs)
	return(wk)

def estimate_wkStar_samples(model, numberOfSamples, min, max, sizes):
	log_wkStar_b = np.zeros(numberOfSamples)
	modelCopy = deepcopy(model)
	# Change the transmission matrix otherwise the model will fit the widest distribution.
	k = modelCopy.transmat_.shape[0]
	modelCopy.transmat_ = 1/k * np.ones((k,k))
	for i in range(numberOfSamples):
		rndSample = np.random.uniform(low=min, high=max, size=sizes)
		log_wkStar_b[i] = np.log(estimate_wk_model(modelCopy, rndSample))
	eStar_log_wkStar = np.mean(log_wkStar_b)
	sk = np.sqrt(1+1/numberOfSamples) * np.std(log_wkStar_b)
	return(eStar_log_wkStar, sk)
