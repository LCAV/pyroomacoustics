Pyroomacoustics
===============

.. image:: https://travis-ci.org/LCAV/pyroomacoustics.svg?branch=pypi-release
    :target: https://travis-ci.org/LCAV/pyroomacoustics
.. image:: https://readthedocs.org/projects/pyroomacoustics/badge/?version=pypi-release
    :target: http://pyroomacoustics.readthedocs.io/en/pypi-release/
    :alt: Documentation Status

Summary
-------

Pyroomacoustics is a software package aimed at the rapid development
and testing of audio array processing algorithms. The content of the package
can be divided into three main components: an intuitive Python object-oriented
interface to quickly construct different simulation scenarios involving
multiple sound sources and microphones in 2D and 3D rooms; a fast C
implementation of the image source model for general polyhedral rooms to
efficiently generate room impulse responses and simulate the propagation
between sources and receivers; and finally, reference implementations of
popular algorithms for beamforming, direction finding, and adaptive filtering.
Together, they form a package with the potential to speed up the time to market
of new algorithms by significantly reducing the implementation overhead in the
performance evaluation step.

Room Acoustics Simulation
`````````````````````````

Consider the following scenario.

  Suppose, for example, you wanted to produce a radio crime drama, and it
  so happens that, according to the scriptwriter, the story line absolutely must culminate
  in a satanic mass that quickly degenerates into a violent shootout, all taking place
  right around the altar of the highly reverberant acoustic environment of Oxford's
  Christ Church cathedral. To ensure that it sounds authentic, you asked the Dean of
  Christ Church for permission to record the final scene inside the cathedral, but
  somehow he fails to be convinced of the artistic merit of your production, and declines
  to give you permission. But recorded in a conventional studio, the scene sounds flat.
  So what do you do?

  -- Schnupp, Nelken, and King, *Auditory Neuroscience*, 2010

Faced with this difficult situation, **pyroomacoustics** can save the day by simulating
the environment of the Christ Church cathedral!

At the core of the package is a room impulse response (RIR) generator based on the
image source model that can handle

* Convex and non-convex rooms
* 2D/3D rooms

Both a pure python implementation and a C accelerator are included for maximum
speed and compatibility.

The philosophy of the package is to abstract all necessary elements of
an experiment using object oriented programming concept. Each of these elements
is represented using a class and an experiment can be designed by combining
these elements just as one would do in a real experiment.

Let's imagine we want to simulate a delay-and-sum beamformer that uses a linear
array with four microphones in a shoe box shaped room that contains only one
source of sound. First, we create a room object, to which we add a microphone
array object, and a sound source object. Then, the room object has methods
to compute the RIR between source and receiver. The beamformer object then extends
the microphone array class and has different methods to compute the weights, for
example delay-and-sum weights. See the example below to get an idea of what the
code looks like.

The `Room` class also allows one to process sound samples emitted by sources,
effectively simulating the propagation of sound between sources and microphones.
At the input of the microphones composing the beamformer, an STFT (short time
Fourier transform) engine allows to quickly process the signals through the
beamformer and evaluate the output.

Reference Implementations
`````````````````````````

In addition to its core image source model simulation, **pyroomacoustics**
also contains a number of reference implementations of popular audio processing
algorithms for

* beamforming
* direction of arrival (DOA) finding
* adaptive filtering (NLMS, RLS)
* blind source separation (AuxIVA, Trinicon)

We use an object-oriented approach to abstract the details of
specific algorithms, making them easy to compare. Each algorithm can be tuned through optional parameters. We have tried to
pre-set values for the tuning parameters so that a run with the default values
will in general produce reasonable results.

Datasets
````````
In an effort to simplify the use of datasets, we provide a few wrappers that
allow to quickly load and sort through some popular speech corpora. At the
moment we support the following.

* `CMU ARCTIC <http://www.festvox.org/cmu_arctic/>`_
* `TIMIT <https://catalog.ldc.upenn.edu/ldc93s1>`_

Quick Install
-------------

Install the package with pip::

    $ pip install pyroomacoustics

The requirements are::

* numpy 
* scipy 
* matplotlib

Example
-------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import pyroomacoustics as pra

    # Create a 4 by 6 metres shoe box room
    room = pra.ShoeBox([4,6])

    # Add a source somewhere in the room
    room.add_source([2.5, 4.5])

    # Create a linear array beamformer with 4 microphones
    # with angle 0 degrees and inter mic distance 10 cm
    R = pra.linear_2D_array([2, 1.5], 4, 0, 0.04) 
    room.add_microphone_array(pra.Beamformer(R, room.fs))

    # Now compute the delay and sum weights for the beamformer
    room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])

    # plot the room and resulting beamformer
    room.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
    plt.show()

Authors
-------

* Robin Scheibler
* Ivan Dokmanić
* Sidney Barthe
* Eric Bezzam
* Hanjie Pan

How to contribute
-----------------

If you would like to contribute, please clone the
`repository <http://github.com/LCAV/pyroomacoustics>`_ and send a pull request.

Academic publications
---------------------

This package was developed to support academic publications. The package
contains implementations for DOA algorithms and acoustic beamformers introduced
in the following papers.

* H\. Pan, R. Scheibler, I. Dokmanic, E. Bezzam and M. Vetterli. *FRIDA: FRI-based DOA estimation for arbitrary array layout*, ICASSP 2017, New Orleans, USA, 2017.
* I\. Dokmanić, R. Scheibler and M. Vetterli. *Raking the Cocktail Party*, in IEEE Journal of Selected Topics in Signal Processing, vol. 9, num. 5, p. 825 - 836, 2015.
* R\. Scheibler, I. Dokmanić and M. Vetterli. *Raking Echoes in the Time Domain*, ICASSP 2015, Brisbane, Australia, 2015.

If you use this package in your own research, please cite `our paper describing it <https://arxiv.org/abs/1710.04196>`_.


  R\. Scheibler, E. Bezzam, I. Dokmanić, *Pyroomacoustics: A Python package for audio room simulations and array processing algorithms*, Proc. IEEE ICASSP, Calgary, CA, 2018.

License
-------

::

  Copyright (c) 2014-2017 EPFL-LCAV

  Permission is hereby granted, free of charge, to any person obtaining a copy of
  this software and associated documentation files (the "Software"), to deal in
  the Software without restriction, including without limitation the rights to
  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
  of the Software, and to permit persons to whom the Software is furnished to do
  so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

