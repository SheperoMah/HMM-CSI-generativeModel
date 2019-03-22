#!/usr/bin/env python3

import numpy as np
import sys
from sklearn.externals import joblib
import auxiliaryFuncs as auxFs
from statsmodels.tsa.stattools import acf



def main(fileName, length=10000):
	model = joblib.load(fileName)
	nStates = model.means_.shape[0]
	X,Z = model.sample(length)
	acfVals = acf(X, nlags=10)
	dictionary = {}
	for i in range(nStates):
		stateDuration = auxFs.find_length_of_consecutive_numbers(Z, i)
		dictionary[i] = np.mean(stateDuration)
	import pdb; pdb.set_trace();

if __name__ == "__main__":
	fileName = sys.argv[1]
	main(fileName)
