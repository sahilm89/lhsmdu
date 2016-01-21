LHS-MDU
--------

This is a package for generating latin hypercube samples with multi-dimensional uniformity.
To use, simply do::

    >>> import lhsmdu 
    >>> k = lhsmdu.sample(2, 20) # Latin Hypercube Sampling with multi-dimensional uniformity 

This will generate a nested list with two variables, with 20 samples each.

To plot and see the difference between Monte Carlo and LHS-MDU sampling for a 2 dimensional system::

    >>> l = lhsmdu.createRandomStandardUniformMatrix(2, 20) # Monte Carlo sampling 
    >>> import matplotlib.pyplot as plt 
    >>> fig = plt.figure() 
    >>> ax = fig.gca()
    >>> ax.set_xticks(numpy.arange(0,1,0.1))
    >>> ax.set_yticks(numpy.arange(0,1,0.1))
    >>> plt.scatter(k[0], k[1], col="g", label="LHS-MDU") 
    >>> plt.scatter(l[0], l[1], col="r", label="MC") 
    >>> plt.grid()
    >>> plt.show() 


