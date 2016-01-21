# To create an orthogonal Latin hypercube with uniform sampling of parameters.
# Author: Sahil Moza
# Date: Jan 20, 2016

''' This is an implementation of Latin Hypercube Sampling with Multi-Dimensional Uniformity (LHS-MDU) from Deutsch and Deutsch, "Latin hypercube sampling with multidimensional uniformity." (2012) 

***Currently only for independent variables***
'''

import numpy as np

import time
start_time = time.time()

numDimensions = 2 ## Number of variables (N)
scalingFactor = 5 ## number > 1 (M) Chosen as 5 as suggested by the paper (above this no improvement.
numSamples = 50 ## Number of samples (L)
numRealizations = scalingFactor*numSamples ## Number of realizations (I)
numToAverage = 2 ## Number of nearest neighbours to average, as more does not seem to add more information (from paper).
randomSeed = 42 ## Seed for the random number generator 
plotFlag = 0

np.random.seed(randomSeed) ## Seeding the random number generator.

def createRandomStandardUniformMatrix(nrow, ncol):
    ''' Creates a matrix with elements drawn from a uniform distribution in [0,1]'''
    rows = [ [np.random.random() for i in range(ncol)] for j in range(nrow)]
    return np.matrix(rows)

def findUpperTriangularColumnDistanceVector(matrix, ncol):
    ''' Finds the 1-D upper triangular euclidean distance vector for the columns of a matrix.'''
    assert ncol == matrix.shape[1]
    distance_1D = []
    for i in range(ncol-1):
        for j in range(i+1,ncol):
            realization_i, realization_j  = matrix[:,i], matrix[:,j]
            distance_1D.append(np.linalg.norm(realization_i - realization_j))
    return distance_1D

def createSymmetricDistanceMatrix(distance, nrow):
    ''' Creates a symmetric distance matrix from an upper triangular 1D distance vector.'''
    distMatrix = np.zeros((nrow,nrow))
    indices = np.triu_indices(nrow,k=1)
    distMatrix[indices] = distance
    distMatrix[(indices[1], indices[0])] = distance # Making symmetric matrix
    return distMatrix

### Creating NxI realization matrix
matrixOfRealizations =  createRandomStandardUniformMatrix(numDimensions, numRealizations)

### Finding distances between column vectors of the matrix to create a distance matrix.
distance_1D = findUpperTriangularColumnDistanceVector(matrixOfRealizations, numRealizations)

## Creating a symmetric IxI distance matrix from the triangular matrix 1D vector.
distMatrix = createSymmetricDistanceMatrix(distance_1D, numRealizations)

## Finding columns from the realization matrix by elimination of nearest neighbours L strata are left.
averageDistance = {i:0 for i in range(numRealizations)}

while(len(averageDistance)>numSamples):
    for rowNum in averageDistance.keys():
        averageDistance.update( {rowNum: np.sum( sorted( distMatrix[rowNum,averageDistance.keys()])[:numToAverage+1])/numToAverage}) # +1 to remove the zero index, appending averageDistance to list
    indexToDelete = min(averageDistance, key=averageDistance.get)
    del averageDistance[indexToDelete]

# Creating the strata matrix to draw samples from.
matrixOfStrata = matrixOfRealizations[:,averageDistance.keys()]

assert numSamples == matrixOfStrata.shape[1]
assert numDimensions == matrixOfStrata.shape[0]

matrixOfSamples = []

# Creating Matrix of Samples from the strata ordering.
for row in range(numDimensions):
    sortedIndicesOfStrata = np.argsort(np.ravel(matrixOfStrata[row,:]))

    # Generating stratified samples
    newSamples =  [ (float(x)/numSamples) + (np.random.random()/numSamples) for x in sortedIndicesOfStrata ]
    matrixOfSamples.append(newSamples)

assert np.min(matrixOfSamples)>=0.
assert np.max(matrixOfSamples)<=1.

print matrixOfSamples

print("--- %s seconds ---" % (time.time() - start_time))

########## Plotting here to show difference between MC sampling and LHS-MDU ###
if plotFlag:

    ############# Generating samples with Monte Carlo (MC) Sampling ###############
    matrixOfRandomSamples =  createRandomUniformMatrix(numDimensions, numSamples)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def plot3D(matrix, color = 'b' ):
        # Plot a 3D scatter plot with the sample points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.ravel(matrix[0,:])
        y = np.ravel(matrix[1,:])
        z = np.ravel(matrix[2,:])
        
        ax.scatter3D( x, y, z, color = color  )
        plt.title("Randomly generated numbers in 3 dimensions")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
    
    def plot2D(matrix, color = 'b'):
        # Plot a 2D scatter plot with the sample points
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.ravel(matrix[0,:])
        y = np.ravel(matrix[1,:])
        ax.scatter( x, y, color = color )
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_xlim((-0.05,1.05))
        ax.set_ylim((-0.05,1.05))
        plt.title("Randomly generated numbers in 2 dimensions")

    matrixOfSamples = np.matrix(matrixOfSamples)
    if numDimensions == 3:
        plot3D(matrixOfSamples)
        plt.show()
        plot3D(matrixOfRandomSamples, color='r')
        plt.show()
    elif numDimensions == 2:
        plot2D(matrixOfSamples)
        plt.show()
        plot2D(matrixOfRandomSamples, color='r')
        plt.show()
    else:
        pass
