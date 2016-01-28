Pyroomacoustics
===============

Pyroomacoustics is a package for audio signal processing for indoor
applications. It was developped as a fast prototyping platform for beamforming
algorithms in indoor scenarios. It contains, among other things, the following
list of modules.

* A small room image source model engine
* An STFT (short time Fourier transform) processor
* Spatialized sound sources and receivers
* A beamformer class for easy prototyping of beamforming algorithms
* Many other useful routines

Dependencies
------------

The package was only tested with Python 2.7.

* Numpy, scipy, matplotlib
* cvxopt (only for one routine in the multirate package)

Academic publications
---------------------

This package was developped to support academic publications. The package contains implementations
for the acoustic beamformers introduced in the following papers.

* I. Dokmanic, R. Scheibler and M. Vetterli. *Raking the Cocktail Party*, in IEEE Journal of Selected Topics in Signal Processing, vol. 9, num. 5, p. 825 - 836, 2015.
* R. Scheibler, I. Dokmanic and M. Vetterli. *Raking Echoes in the Time Domain*, ICASSP 2015, Brisbane, Australia, 2015. 

License
-------

Copyright (c) 2015, LCAV

.. image:: https://i.creativecommons.org/l/by-sa/4.0/88x31.png

pyroomacoustics by `LCAV <http://lcav.epfl.ch>`_ is licensed under a 
`Creative Commons Attribution-ShareAlike 4.0 International License <http://creativecommons.org/licenses/by-sa/4.0/>`_.

Based on a work at http://github.com/LCAV/pyroomacoustics.

