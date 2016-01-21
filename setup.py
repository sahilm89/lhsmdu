from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='lhsmdu',
      version='0.1',
      description='This is an implementation of Latin Hypercube Sampling with Multi-Dimensional Uniformity (LHS-MDU) from Deutsch and Deutsch, "Latin hypercube sampling with multidimensional uniformity.',
      long_description=readme(),
      url='http://github.com/sahilm89/lhsmdu',
      author='Sahil Moza',
      author_email='sahil.moza@gmail.com',
      license='MIT',
      packages=['lhs-mdu'],
      install_requires=[ 'numpy', ],
      zip_safe=False)
