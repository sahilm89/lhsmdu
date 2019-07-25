# To create an orthogonal Latin hypercube with uniform sampling of parameters.
# Author: Sahil Moza
# Date: Jan 21, 2016

''' This is an implementation of Latin Hypercube Sampling with Multi-Dimensional Uniformity (LHS-MDU) from Deutsch and Deutsch, "Latin hypercube sampling with multidimensional uniformity", Journal of Statistical Planning and Inference 142 (2012) , 763-772 

***Currently only for independent variables***
'''

from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.linalg import norm
from numpy import random, matrix, zeros, triu_indices, sum, argsort, ravel, max
from numpy import min as minimum
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen

##### Default variables #####
scalingFactor = 5 ## number > 1 (M) Chosen as 5 as suggested by the paper (above this no improvement.
numToAverage = 2 ## Number of nearest neighbours to average, as more does not seem to add more information (from paper).
randSeed = 42 ## Seed for the random number generator 
random.seed(randSeed) ## Seeding the random number generator.

def setRandomSeed(newRandSeed):
    global randSeed
    randSeed = newRandSeed
    random.seed(randSeed) ## Seeding the random number generator.

def createRandomStandardUniformMatrix(nrow, ncol):
    ''' Creates a matrix with elements drawn from a uniform distribution in [0,1]'''
    rows = [ [random.random() for i in range(ncol)] for j in range(nrow)]
    return matrix(rows)

def findUpperTriangularColumnDistanceVector(inputMatrix, ncol):
    ''' Finds the 1-D upper triangular euclidean distance vector for the columns of a matrix.'''
    assert ncol == inputMatrix.shape[1]
    distance_1D = []
    for i in range(ncol-1):
        for j in range(i+1,ncol):
            realization_i, realization_j  = inputMatrix[:,i], inputMatrix[:,j]
            distance_1D.append(norm(realization_i - realization_j))
    return distance_1D

def createSymmetricDistanceMatrix(distance, nrow):
    ''' Creates a symmetric distance matrix from an upper triangular 1D distance vector.'''
    distMatrix = zeros((nrow,nrow))
    indices = triu_indices(nrow,k=1)
    distMatrix[indices] = distance
    distMatrix[(indices[1], indices[0])] = distance # Making symmetric matrix
    return distMatrix

def eliminateRealizationsToStrata(distance_1D, matrixOfRealizations, numSamples, numToAverage = numToAverage):
    ''' Eliminating realizations using average distance measure to give Strata '''

    numDimensions = matrixOfRealizations.shape[0]
    numRealizations = matrixOfRealizations.shape[1]
    ## Creating a symmetric IxI distance matrix from the triangular matrix 1D vector.
    distMatrix = createSymmetricDistanceMatrix(distance_1D, numRealizations)
 
    ## Finding columns from the realization matrix by elimination of nearest neighbours L strata are left.
    averageDistance = {i:0 for i in range(numRealizations)}
    
    while(len(averageDistance)>numSamples):
        for rowNum in sorted(averageDistance.keys()):
            meanAvgDist = sum( sorted( distMatrix[ rowNum, sorted(averageDistance.keys())])[:numToAverage+1])/numToAverage
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
        setRandSeed(randomSeed)

    ### Number of realizations (I) = Number of samples(L) x scale for oversampling (M)
    numRealizations = scalingFactor*numSamples ## Number of realizations (I)
    ### Creating NxI realization matrix
    matrixOfRealizations =  createRandomStandardUniformMatrix(numDimensions, numRealizations)
    
    ### Finding distances between column vectors of the matrix to create a distance matrix.
    distance_1D = findUpperTriangularColumnDistanceVector(matrixOfRealizations, numRealizations)
    
    ## Eliminating columns from the realization matrix, using the distance measure  to get a strata
    ## matrix with number of columns as number of samples requried.

    global matrixOfStrata
    matrixOfStrata = eliminateRealizationsToStrata(distance_1D, matrixOfRealizations, numSamples)

    matrixOfSamples = resample() 
    
    return matrixOfSamples
