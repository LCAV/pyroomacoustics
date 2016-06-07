Pyroomacoustics
===============

.. image:: https://travis-ci.org/LCAV/pyroomacoustics.svg?branch=master
    :target: https://travis-ci.org/LCAV/pyroomacoustics
.. image:: https://readthedocs.org/projects/pyroomacoustics/badge/?version=latest
    :target: http://pyroomacoustics.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Consider the following scenario.

  Suppose, for example, you wanted to produce a radio crime drama, and it
  so happens that, according to the scriptwriter, the story line absolutely must culminate
  in a satanic mass that quickly degenerates into a violent shootout, all taking place
  right around the altar of the higly reverberant acoustic environment of Oxford's
  Christ Church cathedral. To ensure that it sounds authentic, you asked the Dean of
  Christ Church for permission to record the final scene inside the cathedral, but
  somehow he fails to be convinced of the artistic merit of your production, and declines
  to give you permission. But recorded in a conventional studio, the scene sounds flat.
  So what do you do ?

  -- Schnupp, Nelken, and King, *Auditory Neuroscience*, 2010

Faced with this difficult situation, **pyroomacoustics** can save the day by simulating
the environment of the Christ Church cathedral!

Pyroomacoustics is a pure python package for audio signal processing for indoor
applications. It was developped as a fast prototyping platform for beamforming
algorithms in indoor scenarios. At the core of the package is a room impulse
response generator based on the image source model that can handle

* Convex and non-convex rooms
* 2D/3D rooms

The philosophy of the package is to abstract all necessary elements of
an experiment using object oriented programming concept. Each of these elements
is represented using a class and an experiment can be designed by combining
these elements just as one would do in a real experiment.

Let's imagine we want to simulate a delay and sum beamformer that uses a linear
array with four microphones in a shoe box shaped room that contains only one
source of sound. First, we create a room object, to which we add a microphone
array object, and a sound source object. Then, the room object has methods
to compute the RIR between source and receiver. The beamformer object then extends
the microphone array class and has different methods to compute the weights, for
example delay-and-sum weights. See the example below to get an idea of what the
code looks like.

The `Room` class allows in addition to process sound samples emitted by sources,
effectively simulating the propagation of sound between sources and microphones.
At the input of the microphone composing the beamformer, an STFT (short time
Fourier transform) engine allows to quickly process the signals through the
beamformer and evaluate the ouput.

Quick Install
-------------

The package was only tested with Python 2.7.

* Numpy, scipy, matplotlib
* cvxopt (only for one routine in the multirate package)

Example
-------

.. literalinclude:: ../examples/delay_and_sum.py
   :language: python
   :linenos:

Academic publications
---------------------

This package was developped to support academic publications. The package contains implementations
for the acoustic beamformers introduced in the following papers.

* I\. Dokmanic, R. Scheibler and M. Vetterli. *Raking the Cocktail Party*, in IEEE Journal of Selected Topics in Signal Processing, vol. 9, num. 5, p. 825 - 836, 2015.
* R\. Scheibler, I. Dokmanic and M. Vetterli. *Raking Echoes in the Time Domain*, ICASSP 2015, Brisbane, Australia, 2015. 

License
-------

Copyright (c) 2015, LCAV

.. image:: https://i.creativecommons.org/l/by-sa/4.0/88x31.png

pyroomacoustics by `LCAV <http://lcav.epfl.ch>`_ is licensed under a 
`Creative Commons Attribution-ShareAlike 4.0 International License <http://creativecommons.org/licenses/by-sa/4.0/>`_.

Based on a work at http://github.com/LCAV/pyroomacoustics.

