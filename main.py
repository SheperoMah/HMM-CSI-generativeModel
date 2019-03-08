#!/usr/bin/env python3

import sys
import math
import numpy as np
import scipy.stats as stts
import auxiliaryFuncs as auxFs
from statsmodels.tsa.stattools import acf
from sklearn.externals import joblib

def divide_data(data):
	""" Divides data into training, validation, and test data sets. The division ratio depends on the variable ratio which should be of length 2. Training data are placed in rows.
	"""
	#assert max(ratio) < 1.0 and min(ratio) > 0.0 and len(ratio) == 2, "The variable ratio should be of length 2, where the first term encodes the ratio of the training data, the second term encodes the ratio of the validataion data. The remaining is the test data. All values (0,1)."
	numDataPoints = data.shape[0]
	indexes = np.arange(numDataPoints)
	#np.random.shuffle(indexes)
	#part = [math.floor(i*numDataPoints) for i in ratio]
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

def main(filename, nStates, seed):
	ds = np.loadtxt(f"{filename}.txt").transpose()
	trDs, valDs = divide_data(ds)
	lengthTr = calculate_length_sequence(trDs)
	lengthVal = calculate_length_sequence(valDs)
	print(len(lengthTr), len(lengthVal))
	trDs = trDs.reshape(-1,1)
	valDs = valDs.reshape(-1,1)
	dataSets = [trDs, valDs]
	lengths = [lengthTr, lengthVal]
	model = auxFs.train_hmm_model(trDs, lengthTr, nStates, 1, 400, 1e-3, seed)
	scores = score_model_on_datasets(model, dataSets, lengths)
	print("log-likelihood, ", scores)
	ksScores = ks_test(model, dataSets)
	print("ksScores, ", ksScores)
	joblib.dump(model, f"{filename}_{nStates}_100.pkl")
	oldMatrix = model.transmat_
	fact = [0.8, 0.4, 0.3, 0.2]
	for i in fact:
		newModel = auxFs.change_transition_matrix_of_model(model, oldMatrix, i)
		scores = score_model_on_datasets(newModel, dataSets, lengths)
		print("log-likelihood, ", scores)
		ksScores = ks_test(newModel, dataSets)
		print("ksScores, ", ksScores)
		joblib.dump(newModel, f"{filename}_{nStates}_{i*100:.0f}.pkl")

	#import pdb; pdb.set_trace()


if __name__ == "__main__":
	filename = sys.argv[1]
	nStates = int(sys.argv[2])
	seed = int(sys.argv[3])
	main(filename, nStates, seed)
