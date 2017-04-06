Pyroomacoustics
===============

.. image:: https://travis-ci.org/LCAV/pyroomacoustics.svg?branch=pypi-release
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
    room1 = pra.ShoeBox([4,6])

    # Add a source somewhere in the room
    room1.addSource([2.5, 4.5])

    # Create a linear array beamformer with 4 microphones
    # with angle 0 degrees and inter mic distance 10 cm
    R = pra.linear2DArray([2, 1.5], 4, 0, 0.04) 
    room1.addMicrophoneArray(pra.Beamformer(R, room1.fs))

    # Now compute the delay and sum weights for the beamformer
    room1.micArray.rakeDelayAndSumWeights(room1.sources[0][:1])

    # plot the room and resulting beamformer
    room1.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
    plt.show()

Authors
-------

* Robin Scheibler
* Ivan Dokmanić
* Sidney Barthe

How to contribute
-----------------

If you would like to contribute, please clone the
`repository <http://github.com/LCAV/pyroomacoustics>`_ and send a pull request.

Academic publications
---------------------

This package was developped to support academic publications. The package contains implementations
for the acoustic beamformers introduced in the following papers.

* I\. Dokmanic, R. Scheibler and M. Vetterli. *Raking the Cocktail Party*, in IEEE Journal of Selected Topics in Signal Processing, vol. 9, num. 5, p. 825 - 836, 2015.
* R\. Scheibler, I. Dokmanic and M. Vetterli. *Raking Echoes in the Time Domain*, ICASSP 2015, Brisbane, Australia, 2015. 

License
-------

::

  Copyright (c) 2014-2017, LCAV

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

