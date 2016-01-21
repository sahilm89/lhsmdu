''' This is a benchmark of the amount of time the algorithm takes to spit out LHS samples'''
from time import time
import lhsmdu

def runtime(numDimensions, numSamples):
    ''' Checks runtime using standard variables '''
    start_time = time()
    m = lhsmdu.sample(numDimensions,numSamples)
    end_time = time()
    print end_time-start_time
