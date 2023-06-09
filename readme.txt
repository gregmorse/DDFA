-------------
1. LICENSE
-------------

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License version 2 as published
by the Free Software Foundation (LGPL may be granted upon request). This program is
distributed in the hope that it will be useful, but without any warranty; without even
the implied warranty of merchantability or fitness for a particular purpose. See the
GNU General Public License for more details.

---------------------
2. USAGE
---------------------

INTRODUCTION
---------------------
This package implements the DDFA algorithm as described in 
http://eplex.cs.ucf.edu/papers/szerlip_aaai15.pdf and is intended to provide a
useful starting point for those interested in either modifying the DDFA
algorithm or applying it to new domains.  The DDFA algorithm is built atop HyperNEAT,
which is described in http://eplex.cs.ucf.edu/papers/gauci_gecco07.pdf and a good
understanding of HyperNEAT is recommended before attempting to modify or extend DDFA.

The DDFA alogrithm can be broken up into two steps.  The first step is the discovery
of novel features through a process we will refer to as "feature collection".  
The second step is composition of a set of features into a neural network 
and training of that network through backpropagation.

SOFTWARE INFO
---------------------
The software package is written in C# and is built upon SharpNeat, which is a NEAT 
package released by Colin Green.  The package consists of two folders: 
- DFN contains a Visual Studio 2010 solution (DFN.sln) and all projects/code is 
contained within this solution
- data contains the sample data for the MNIST and CIFAR-100 data sets

RUNNING DDFA
---------------------
To run the program in feature collection mode, execute the program with a single 
command line argument of "collection".  In feature collection mode, the program will 
generate a log file called archivesize.txt that tracks the number of novel features 
discovered.  In addition, a folder will be generated called exp_#_##_##_##_#/novelty 
that contains the discovered features in xml format.
Once the desired number of features has been collected, the program may be stopped.

To run the program in backpropagation mode, execute the program with a single command 
line argument of "backprop".  The desired features to use should be placed in a folder 
called "archive".  The features in the archive folder will be 
composed into a neural network with one hidden layer of neurons, with the weights 
connecting from the hidden layer to the output layer being randomized.
A multithreaded implementation of the backpropagatation algorithm is used by default.

PARAMETER MODIFICATION
---------------------
Many parameters for both the collection and backprop steps are available to tune 
through the class DFNParams.  As the DDFA algorithm is highly parallelizable, 
it is recommended that THREADS is set to the number of cores available on your 
machine.  Note that performance of the multithreaded backprop implementation may 
change based on the number of threads.  We found using more than 8 threads for 
backprop led to worse performance.


ADDING A NEW DOMAIN
---------------------
To add a new domain, perform the following steps:
1) Add a class that implements the IDomain interface.  For an example of this, 
see the MNISTManager class.
2) Add a class that implements the IDomainEvaluator interface.  For an example of this,
see the MNISTEvaluator class.
3) Add a class for generating the CPPN substrate.  For an example of this, see the
MNISTSubstrateDecoder class.
4) Add a new domain to the Domain enum in the DFNParams class and set the DOMAIN 
parameter to your new domain.
5) In the Program class, there are two places where DomainManager.Instance is set.  
Add code to set this to your IDomain.Instance that was created in step 1.


SUPPORT
---------------------
For all questions, comments, bug reports, suggestions, or friendly hellos, 
send email to GregoryMorse07@gmail.com


DATA
---------------------
CIFAR-10 data can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
MNIST data can be downloaded from http://yann.lecun.com/exdb/mnist/
