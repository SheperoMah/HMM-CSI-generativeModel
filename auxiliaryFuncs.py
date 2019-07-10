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
    """ Changes the transition matrix of the HMM model. It scales the
    off-diagonal elements of the randomMat, and renormalize it, then
    returns a new model with the scaled transition matrix. c has to be in
    range (0,1].
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

def train_hmm_model_fixed_means(trainingData, lengthSequences, numHiddenStates, dimObservations, numIterations, tolerance, randomSeed, means):
    """ Trains a Gaussian Mixture HMM model with fixed means.

    It initializes the transmission matrix, and covariances
    """
    np.random.seed(randomSeed)
    model = hmm.GaussianHMM(n_components=numHiddenStates, init_params="stc",
                params="stc", n_iter=numIterations, tol=tolerance)
    model.n_features = dimObservations
    assert means.shape == (model.n_components, model.n_features), \
        "The means array should have shape (n_components, n_features)"
    model.means_ = means
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

def sample_model(model, dataSets):
    modelSamples = [model.sample(len(v))[0] for i, v in enumerate(dataSets)]
    return(modelSamples)

def ks_test(modelSample, dataSets):
    scores = [stts.ks_2samp(modelSample.flatten(), i.flatten()).statistic for i in dataSets]
    return(scores)

def estimate_pdf_using_histogram(data, binList):
     apdf, _ = np.histogram(data, bins=binList, density=True)
     return(apdf)

#TODO correct this function
def kld_test(modelSample, dataSets, points=np.linspace(0, 1.5, 25)):
    epdfDataSets = [estimate_pdf_using_histogram(i, points) for i in dataSets]
    epdfSamples = estimate_pdf_using_histogram(modelSample.flatten(), points)
    scores = [stts.entropy(i.flatten(), epdfSamples) for i in epdfDataSets]
    return(scores)

def estimate_aic_score(logLik, n, lambda_k):
    d = n * lambda_k + n * (n - 1)
    return(-2*logLik + d)

def estimate_bic_score(logLik, n, lambda_k, T):
    d = n * lambda_k + n * (n - 1)
    return(-2 * logLik + np.log(T) * d)

def clean_axes(ax):
    """ Removes the right and top axes of matplotlib
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def sample_markov_chain(transMatrix, length, initialState=0):
    """ Samples a state trajectory from a Markov chain.

    Paramters
    ---------
    transMatrix : ndarray
        The Markov chain transition matrix. Must be of shape N * N, 
    where N is the number of states.
    length : int
        Length of the sample, i.e., number of time-steps.
    initialState : int
        The initial state, It must be less than 0 <= initialState < N.

    Returns
    -------
    list
        A list with the index of state at each time step.
    """
    assert transMatrix.shape[0] == transMatrix.shape[1], "Transition "+\
    "matrix must be square."
    assert initialState < transMatrix.shape[0] and initialState >= 0 \
    and type(initialState) is int, "The initialState must be integer "+\
    "and 0 <= initialState < numberOfStates."
    assert length > 0 and type(length) is int, "Length must be " + \
    "positive integer."
    
    rndSmpl = np.random.uniform(size=length)
    cumSumTransMatrix = np.cumsum(transMatrix, axis=1) 
    states = [initialState]

    for i in range(length):
        currentState = states[-1]
        cSTMRow = cumSumTransMatrix[currentState,:]
        nextState = np.where(cSTMRow >= rndSmpl[i])[0][0]        
        states.extend([nextState])
    return(states)

def sample_observations(stateSeq, dist, limitPositive=False):
    """ Samples from distributions with each belonging to state.
    
    Examples
    --------
    >>> dist = {0: lambda: np.random.normal(0.4,1), \
                1: lambda: np.random.normal(3,0.2)}
    >>> states = [0,1,0,1,0,0,0,1,1,0,1,1,1,1]
    >>> sample_observations(stateSeq, dist)
    >>> sample_observations(stateSeq, dist, True)

    """
    length = len(stateSeq)
    samples = []
    for i in stateSeq:
        newSample = dist[i]()
    
        while limitPositive and newSample < 0.0:
            newSample = dist[i]()
      
        samples.extend([newSample])
    return(samples)

def sample_hmm_model(model, length, initialState, limitPositive=False):
    """ Samples from the HMM model. 
    """
    hiddenStates = sample_markov_chain(model.transmat_, 
                        length, initialState)
    def create_distribution_dict(means, variances):
        dist={}
        for i in range(means.shape[0]):
            dist[i] = lambda x=i: np.random.normal(means[x], 
                            np.sqrt(variances[x]))
        return(dist)
    dist = create_distribution_dict(model.means_.flatten(), 
                model.covars_.flatten())
    observations = sample_observations(hiddenStates, dist, 
                        limitPositive) 
    return(np.array(observations), np.array(hiddenStates))

def ACF(ts, numLags):
    """ Estimates the ACF for a time series, works only for unidimentional time series.

    """
    average = np.mean(ts)
    variance = np.var(ts)
    def acf_one_point(ts, mean, variance, h):
        length = ts.shape[0]
        acf = np.mean((ts[0:length - h] - mean) * (ts[h::] - mean)) / variance
        return(acf)
    return([acf_one_point(ts, average, variance, i) for i in range(numLags+1)])


def find_length_of_consecutive_numbers(data, number):
    """ Returns the length of the sequence of number in the array.
    >>> data = np.array([1, 1, 1, 0, 1, 0, 0, 1, 1])
    >>> find_length_of_consecutive_numbers(data, 1)
    np.array([3, 1, 2])
    """
    dataOnes = np.ones(data.shape) * (data == number)
    dataWithInitZero = np.insert(dataOnes, 0, 0.0)
    dataWithEndZero = np.append(dataWithInitZero, 0.0)
    diffData = np.diff(dataWithEndZero)
    lengths = np.where(diffData == -1)[0] - np.where(diffData == 1)[0]
    return(lengths)


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
