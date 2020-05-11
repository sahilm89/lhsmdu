# To create an orthogonal Latin hypercube with uniform sampling of parameters.
# Author: Sahil Moza
# Date: Jan 21, 2016

''' This is an implementation of Latin Hypercube Sampling with Multi-Dimensional Uniformity (LHS-MDU) from Deutsch and Deutsch, "Latin hypercube sampling with multidimensional uniformity", Journal of Statistical Planning and Inference 142 (2012) , 763-772 

***Currently only for independent variables***
'''

from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.linalg import norm
from numpy import random, matrix, zeros, triu_indices, sum, sort, argsort, ravel, max, min as minimum, mean, argpartition 
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen
from scipy.spatial.distance import pdist, squareform

##### Default variables #####
scalingFactor = 5 ## number > 1 (M) Chosen as 5 as suggested by the paper (above this no improvement.
numToAverage = 2 ## Number of nearest neighbours to average, as more does not seem to add more information (from paper).
randSeed = 42 ## Seed for the random number generator 
random.seed(randSeed) ## Seeding the random number generator.
matrixOfStrata = [] 

def setRandomSeed(newRandSeed):
    global randSeed
    randSeed = newRandSeed
    random.seed(randSeed) ## Seeding the random number generator.

def eliminateRealizationsToStrata(distance_1D, matrixOfRealizations, numSamples, numToAverage = numToAverage):
    ''' Eliminating realizations using average distance measure to give Strata '''

    numDimensions = matrixOfRealizations.shape[0]
    numRealizations = matrixOfRealizations.shape[1]
    ## Creating a symmetric IxI distance matrix from the triangular matrix 1D vector.
    distMatrix = squareform(distance_1D)

    ## Finding columns from the realization matrix by elimination of nearest neighbours L strata are left.
    averageDistance = {i:0 for i in range(numRealizations)}

    while(len(averageDistance)>numSamples):
        realizations = sort(averageDistance.keys())
        for rowNum in realizations:
            idx = realizations[argpartition(distMatrix[rowNum, realizations], numToAverage-1)[:numToAverage]] # Argpartition kth is numToAverage-1 because it is the index of the in-position sorted sequence (which starts at 0.
            meanAvgDist = mean(distMatrix[rowNum, idx])
            averageDistance.update( {rowNum: meanAvgDist }) # +1 to remove the zero index, appending averageDistance to list
        indexToDelete = min(averageDistance, key=averageDistance.get)
        del averageDistance[indexToDelete]
    
    # Creating the strata matrix to draw samples from.
    StrataMatrix = matrixOfRealizations[:,sorted(averageDistance.keys())]

    assert numSamples == StrataMatrix.shape[1]
    assert numDimensions == StrataMatrix.shape[0]
    #print ( StrataMatrix )
    return StrataMatrix

def inverseTransformSample(distribution, uniformSamples):
    ''' This function lets you convert from a standard uniform sample [0,1] to
    a sample from an arbitrary distribution. This is done by taking the cdf [0,1] of 
    the arbitrary distribution, and calculating its inverse to picking the sample."
    '''
    assert (isinstance(distribution, rv_continuous) or isinstance(distribution, rv_discrete) or isinstance(distribution,rv_frozen))
    newSamples = distribution.ppf(uniformSamples)
    return newSamples
    
def resample():
    ''' Resampling function from the same strata'''
    if not len(matrixOfStrata):
        raise Exception(matrixOfStrata, "Empty strata matrix")
    numDimensions = matrixOfStrata.shape[0]
    numSamples = matrixOfStrata.shape[1]

    matrixOfSamples = []
    
    # Creating Matrix of Samples from the strata ordering.
    for row in range(numDimensions):
        sortedIndicesOfStrata = argsort(ravel(matrixOfStrata[row,:]))
    
        # Generating stratified samples
        newSamples =  [ (float(x)/numSamples) + (random.random()/numSamples) for x in sortedIndicesOfStrata ]
        matrixOfSamples.append(newSamples)
    
    assert minimum(matrixOfSamples)>=0.
    assert max(matrixOfSamples)<=1.
    
    return matrix(matrixOfSamples)

def sample(numDimensions, numSamples, scalingFactor=scalingFactor, numToAverage = numToAverage, randomSeed=randSeed):
    ''' Main LHS-MDU sampling function '''

    if not randomSeed == randSeed:
        setRandomSeed(randomSeed)

    ### Number of realizations (I) = Number of samples(L) x scale for oversampling (M)
    numRealizations = scalingFactor*numSamples ## Number of realizations (I)
    ### Creating NxI realization matrix
    matrixOfRealizations =  random.uniform(size=(numDimensions, numRealizations))
    
    ### Finding distances between column vectors of the matrix to create a distance matrix.
    distance_1D = pdist(matrixOfRealizations.T)

    ## Eliminating columns from the realization matrix, using the distance measure  to get a strata matrix with number of columns as number of samples requried.

    global matrixOfStrata
    matrixOfStrata = eliminateRealizationsToStrata(distance_1D, matrixOfRealizations, numSamples)
    matrixOfSamples = resample() 
    
    return matrixOfSamples
