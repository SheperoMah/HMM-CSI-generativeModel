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

def estimate_CSI_bar(ts, cutoff):
    """ Estimates the k_bar, used in HOLLANDS, KG Terry; SUEHRCKE, Harry. 
    A three-state model for the probability distribution of 
    instantaneous solar radiation, with applications. Solar Energy, 
    2013, 96: 103-112.

    """
    S = np.sum(ts>=cutoff) / np.sum(ts>=0.0)    
    return(np.sqrt(S))


def main(filename, seed, sampleLength=1000000):
    ds = np.loadtxt(f"{filename}.txt").transpose()
    trDs, valDs = auxFs.divide_data(ds)
    lengthTr = auxFs.calculate_length_sequence(trDs)
    lengthVal = auxFs.calculate_length_sequence(valDs)
    trDs = trDs.reshape(-1,1)
    valDs = valDs.reshape(-1,1)
    dataSets = [trDs, valDs]
    lengths = [lengthTr, lengthVal]
    scores = {'seed': seed}
    nStates = 3
    kBar = estimate_CSI_bar(ds, 0.95) 
    thirdMean = 0.1205 + (0.3341*kBar)
    print("Mean of broken: ", thirdMean)
    means = np.array([1.04, 1.00, thirdMean]).reshape(3,1)
    tStart = time.time()
    model = auxFs.train_hmm_model_fixed_means(trDs, lengthTr, nStates, 1, 400,
                 1e-3, seed, means)
    tEnd = time.time()
    joblib.dump(model, f"{filename}_{nStates}_100_fixedMeans.pkl")
    scores['train_time_100'] = math.ceil(tEnd - tStart)
    scores['likelihood_100'] = auxFs.score_model_on_datasets(model, dataSets, lengths)
    modelSample, _  = auxFs.sample_hmm_model(model, sampleLength, 0, True)
    np.savetxt(f"{filename}_{nStates}_100_fixedMeans.txt", modelSample)
    scores['K-S_100'] = auxFs.ks_test(modelSample, dataSets)
    scores['aic_100'] = auxFs.estimate_aic_score(scores['likelihood_100'][0], nStates, 2)
    scores['bic_100'] = auxFs.estimate_bic_score(scores['likelihood_100'][0], nStates, 2, sum(lengthTr))
    oldMatrix = model.transmat_

    if nStates == 3:
        fact = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
        for i in fact:
            locStr = f'_{i*100:0.0f}'
            newModel = auxFs.change_transition_matrix_of_model(model, oldMatrix, i)
            scores[f'likelihood{locStr}'] = auxFs.score_model_on_datasets(newModel, dataSets, lengths)
            modelSample, _ = auxFs.sample_hmm_model(newModel, 
                                sampleLength, 0, True)
            np.savetxt(f"{filename}_{nStates}{locStr}_fixedMeans.txt", modelSample)
            scores[f'K-S{locStr}'] = auxFs.ks_test(modelSample, dataSets)
            scores[f'aic{locStr}'] = auxFs.estimate_aic_score(scores[f'likelihood{locStr}'][0], nStates, 2)
            scores[f'bic{locStr}'] = auxFs.estimate_bic_score(scores[f'likelihood{locStr}'][0], nStates, 2, sum(lengthTr))
            joblib.dump(newModel, f"{filename}_{nStates}{locStr}_fixedMeans.pkl")

    with open(f"{filename}_{nStates}_{seed}_fixedMeans.json", 'w') as f:
        json.dump(scores, f, indent=4)
    #import pdb; pdb.set_trace()


if __name__ == "__main__":
    filename = sys.argv[1]
    seed = int(sys.argv[2])
    main(filename, seed)
