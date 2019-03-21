#!/usr/bin/env python3

import sys
import math
import time
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
	for nStates in range(3,4):
		tStart = time.time()
		model = auxFs.train_hmm_model(trDs, lengthTr, nStates, 1, 400, 1e-3, seed)
		tEnd = time.time()
		joblib.dump(model, f"{filename}_{nStates}_100.pkl")
		scores['train_time_100'] = math.ceil(tEnd - tStart)
		scores['likelihood_100'] = auxFs.score_model_on_datasets(model, dataSets, lengths)
		modelSamples = auxFs.sample_model(model, dataSets)
		scores['K-S_100'] = auxFs.ks_test(modelSamples, dataSets)
		scores['KLD_100'] = auxFs.kld_test(modelSamples, dataSets)
		scores['aic_100'] = auxFs.estimate_aic_score(scores['likelihood_100'][0], nStates, 2)
		scores['bic_100'] = auxFs.estimate_bic_score(scores['likelihood_100'][0], nStates, 2, sum(lengthTr))
		oldMatrix = model.transmat_

		fact = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
		for i in fact:
			locStr = f'_{i*100:0.0f}'
			newModel = auxFs.change_transition_matrix_of_model(model, oldMatrix, i)
			scores[f'likelihood{locStr}'] = auxFs.score_model_on_datasets(newModel, dataSets, lengths)
			modelSamples = auxFs.sample_model(newModel, dataSets)
			scores[f'K-S{locStr}'] = auxFs.ks_test(modelSamples, dataSets)
			scores[f'KLD{locStr}'] = auxFs.kld_test(modelSamples, dataSets)
			scores[f'aic{locStr}'] = auxFs.estimate_aic_score(scores[f'likelihood{locStr}'][0], nStates, 2)
			scores[f'bic{locStr}'] = auxFs.estimate_bic_score(scores[f'likelihood{locStr}'][0], nStates, 2, sum(lengthTr))
			joblib.dump(newModel, f"{filename}_{nStates}{locStr}.pkl")

		with open(f"{filename}_{nStates}_{seed}.json", 'w') as f:
			json.dump(scores, f, indent=4)
	#import pdb; pdb.set_trace()


if __name__ == "__main__":
	filename = sys.argv[1]
	seed = int(sys.argv[2])
	main(filename, seed)
