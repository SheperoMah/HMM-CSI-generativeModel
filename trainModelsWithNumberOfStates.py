#!/usr/bin/env python3

import sys
import math
import numpy as np
import scipy.stats as stts
import auxiliaryFuncs as auxFs
from statsmodels.tsa.stattools import acf
from sklearn.externals import joblib
import json

def main(filename, seed):
	ds = np.loadtxt(f"{filename}.txt").transpose()	
	trDs, valDs = auxFs.divide_data(ds)
	lengthTr = auxFs.calculate_length_sequence(trDs)
	lengthVal = auxFs.calculate_length_sequence(valDs)
	trDs = trDs.reshape(-1,1)
	valDs = valDs.reshape(-1,1)
	dataSets = [trDs, valDs]
	lengths = [lengthTr, lengthVal]
	scores = {'seed': seed}
	for nStates in range(2,13):
		model = auxFs.train_hmm_model(trDs, lengthTr, nStates, 1, 400, 1e-3, seed)
		joblib.dump(model, f"{filename}_{nStates}_100.pkl")
		scores['likelihood_100'] = auxFs.score_model_on_datasets(model, dataSets, lengths)
		scores['K-S_100'] = auxFs.ks_test(model, dataSets)
		oldMatrix = model.transmat_	
		
		fact = [0.8, 0.4, 0.3, 0.2]
		for i in fact:
			newModel = auxFs.change_transition_matrix_of_model(model, oldMatrix, i)
			scores[f'likelihood_{i*100:0.0f}'] = auxFs.score_model_on_datasets(newModel, dataSets, lengths)
			scores[f'K-S_{i*100:0.0f}'] = auxFs.ks_test(newModel, dataSets)
			joblib.dump(newModel, f"{filename}_{nStates}_{i*100:.0f}.pkl")
		
		with open(f"{filename}_{nStates}_{seed}.json", 'w') as f:
			json.dump(scores, f, indent=4)
	#import pdb; pdb.set_trace()


if __name__ == "__main__":
	filename = sys.argv[1]
	seed = int(sys.argv[2])
	main(filename, seed)


