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

* I. Dokmanic, R. Scheibler and M. Vetterli. _Raking the Cocktail Party_, in IEEE Journal of Selected Topics in Signal Processing, vol. 9, num. 5, p. 825 - 836, 2015.
* R. Scheibler, I. Dokmanic and M. Vetterli. _Raking Echoes in the Time Domain_, ICASSP 2015, Brisbane, Australia, 2015. 

Contributors
------------

Ivan Dokmanić, Robin Scheibler, Sidney Barthe.

License
-------

Copyright (c) 2014, Ivan Dokmanić, Robin Scheibler.

This code is free to reuse for non-commercial purpose such as academic or
educational. For any other use, please contact the authors.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">pyroomacoustics</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://lcav.epfl.ch" property="cc:attributionName" rel="cc:attributionURL">Ivan Dokmanić, Robin Scheibler, Sidney Barthe</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/LCAV/pyroomacoustics" rel="dct:source">https://github.com/LCAV/pyroomacoustics</a>.
