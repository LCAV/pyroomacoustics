Changelog
=========

All notable changes to `pyroomacoustics
<https://github.com/LCAV/pyroomacoustics>`_ will be documented in this file.

The format is based on `Keep a
Changelog <http://keepachangelog.com/en/1.0.0/>`__ and this project
adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.

`Unreleased`_
-------------

Added
~~~~~

- New global parameters to control the octave bands used for simulation.
  * ``octave_bands_base_freq``: the base frequency used for the octave bands (default ``125 ``),
    note that together with the sampling frequency this will determine the number of sub-bands
    used in simulation.
  * ``octave_bands_n_fft``: lengths of the octave band filters (current is default ``512``
    but will be changed to ``128`` in the next release)
  * ``octave_bands_keep_dc``: extends the lowest band to include the DC offset,
    the current default is ``False`` to match past behavior, but will be changed to
    ``True`` in the next release because the filters have less oscillations this way.
- Simulation with measured directivity responses in SOFA format (limited file types) is
  possible with the image source model.
- Adds `soxr <https://github.com/dofuuz/python-soxr>`_ as a dependency since resampling
  can often be necessary when working with SOFA files.

Changed
~~~~~~~

- In ray tracing, the histograms are now linearly interpolated between
  the bins to obtain smoother RIR
- Extra parameter ``energy_thresh`` added to ``pyroomacoustics.experimental.measure_rt60``.
  The energy tail beyond this threshold is discarded which is useful for noisy RIR
  measurements. The default value is 0.95.

Bugfix
~~~~~~

- Fixes a bug in ``beamforming.circular_microphone_array_xyplane`` (#348)
- Fixes a bug in ``room.Room.plot_rir`` where the x-axis would have the wrong
  ticks when called with ``kind="tf"``

`0.7.4`_ - 2024-04-25
---------------------

Added
~~~~~

- New implementation of fast RIR builder function in the ``libroom`` C++
  extension to replace the current cython code. Advantages are: 1) only one
  compiled extension, 2) multithreading support
- New global parameter ``sinc_lut_granularity`` that controls the number of
  points used in the look-up table for the sinc interpolation. Accessible via
  ``parameters.constants.get``.
- New global parameter  ``num_threads`` that controls the number of threads
  used in multi-threaded code (rir builder only at the moment). The number of
  threads can also be controlled via the environement variable
  ``PRA_NUM_THREADS``
- Adds package build support for Python 3.11 and 3.12. - Adds package build for
  new Apple M1 architecture

Changed
~~~~~~~

- Removed the broken ``get_rir`` method of the class ``SoundSource``
- Removes package build support for Python 3.7 (EOL)

Bugfix
~~~~~~

- Fixes a bug when using randomized image source model with a 2D room (#315) by
  @hrosseel
- Fixes a bug when setting the air absorption coefficients to custom values (#191),
  adds a test, and more details in the doc
- Fixes a bug in the utilities.angle_function in the calculation of the
  colatitude (#329) by @fabiodimarco
- Replaces the crossing-based point-in-polygon algorithm in the C++ code with
  the more robust winding number algorithm (#345)
- Fixes usage of deprecated hann window with new version of scipy in
  `metrics.py` (#344) by @mattpitkin

`0.7.3`_ - 2022-12-05
---------------------

Bugfix
~~~~~~

- Fixes issue #293 due to the C++ method ``Room::next_wall_hit`` not handling
  2D shoebox rooms, which cause a seg fault

`0.7.2`_ - 2022-11-15
---------------------

Added
~~~~~

- Appveyor builds for compiled wheels for win32/win64 x86

Bugfix
~~~~~~

- Fixes missing import statement in room.plot for 3D rooms (PR #286)
- On win64, ``bss.fastmnmf`` would fail due to some singular matrix. 1) protect solve
  with try/except and switch to pseudo-inverse if necessary, 2) change eps 1e-7 -> 1e-6

`0.7.1`_ - 2022-11-11
---------------------

Bugfix
~~~~~~

- Fixed pypi upload for windows wheels

`0.7.0`_ - 2022-11-10
---------------------

Added
~~~~~

- Added the AnechoicRoom class.
- Added FastMNMF2 (Fast Multichannel Nonnegative Matrix Factorization 2) to ``bss`` subpackage.
- Randomized image source method for removing sweeping echoes in shoebox rooms.
- Adds the ``cart2spher`` method in ``pyroomacoustics.doa.utils`` to convert from cartesian
  to spherical coordinates.
- Example `room_complex_wall_materials.py`
- CI for python 3.10

Changed
~~~~~~~

- Cleans up the plot_rir function in Room so that the labels are neater. It
  also adds an extra option ``kind`` that can take values "ir", "tf", or "spec"
  to plot the impulse responses, transfer functions, or spectrograms of the RIR.
- Refactored the implementation of FastMNMF.
- Modified the document of __init__.py in ``doa`` subpackage.
- `End of Python 3.6 support <https://endoflife.date/python>`__.
- Removed the deprecated ``realtime`` sub-module.
- Removed the deprecated functions ``pyroomacoustics.transform.analysis``, ``pyroomacoustics.transform.synthesis``, ``pyroomacoustics.transform.compute_synthesis_window``. They are replaced by the equivalent functions in ``pyroomacoustics.transform.stft`` sub-module.
- The minimum required version of numpy was changed to 1.13.0 (use of ``np.linalg.multi_dot`` in ``doa`` sub-package see #271)

Bugfix
~~~~~~

- Fixed most warnings in the tests
- Fixed bug in ``examples/adaptive_filter_stft_domain.py``

`0.6.0`_ - 2021-11-29
---------------------

Added
~~~~~

- New DOA method: MUSIC with pseudo-spectra normalization. Thanks @4bian!
  Normalizes MUSIC pseudo spectra before averaging across frequency axis.

Bugfix
~~~~~~

- Issue 235: fails when set_ray_tracing is called, but no mic_array is set
- Issue 236: general ISM produces the wrong transmission coefficients
- Removes an unncessery warning for some rooms when ray tracing is not needed

Misc
~~~~

- Unify code format by using Black
- Add code linting in continuous integration
- Drop CI support for python 3.5


`0.5.0`_ - 2021-09-06
---------------------

Added
~~~~~

- Adds tracking of reflection order with respect to x/y/z axis in the shoebox image
  source model engine. The orders are available in `source.orders_xyz` after running
  the image source model
- Support for microphone and source directivites for image source model. Source
  directivities just for shoebox room. Available directivities are frequency-independent
  (cardioid patterns), although the infrastructure is there for frequency-dependent
  directivities: frequency-dependent usage in `Room.compute_rir` and abstract
  `Directivity` class.
- Examples scripts and notebook for directivities.

Bugfix
~~~~~~

- Fix wrong bracketing for negative values in is_inside (ShoeBox)

`0.4.3`_ - 2021-02-18
---------------------

Added
~~~~~

- Support for Python 3.8 and 3.9

Bugfix
~~~~~~

- Fixes typo in a docstring
- Update docs to better reflect actual function parameters
- Fixes the computation of the cost function of SRP-PHAT doa algorithm (bug reported in #PR197)

Changed
~~~~~~~

- Improve the computation of the auxiliary variables in AuxIVA and ILRMA.
  Unnecessary division operations are reduced.

`0.4.2`_ - 2020-09-24
---------------------

Bugfix
~~~~~~

- Fixes the Dockerfile so that we don't have to install the build dependencies manually
- Change the eps for geometry computations from 1e-4 to 1e-5 in ``libroom``

Added
~~~~~

- A specialized ``is_inside`` routine for ``ShoeBox`` rooms

`0.4.1`_ - 2020-07-02
---------------------

Bugfix
~~~~~~

- Issue #162 (crash with max_order>31 on windows), seems fixed by the new C++ simulator
- Test for issue #162 added
- Fix Binder link
- Adds the pyproject.toml file in MANIFEST.in so that it gets picked up for packaging

Added
~~~~~

- Minimal `Dockerfile` example.

`0.4.0`_ - 2020-06-03
---------------------

Improved Simulator with Ray Tracing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Ray Tracing in the libroom module. The function compute_rir() of the Room object in python
  can now be executed using a pure ray tracing approach or a hybrid (ISM + RT) approach.
  That's why this function has now several default arguments to run ray tracing (number
  of rays, scattering coefficient, energy and time thresholds, microphone's radius).
- Bandpass filterbank construction in ``pyroomacoustics.acoustics.bandpass_filterbank``
- Acoustic properties of different materials in ``pyroomacoustics.materials``
- Scattering from the wall is handled via ray tracing method, scattering coefficients are provided
  in ``pyroomacoustics.materials.Material`` objects
- Function ``inverse_sabine`` allows to compute the ``absorption`` and ``max_order`` to use with
  the image source model to achieve a given reverberation time
- The method ``rt60_theory`` in ``pyroomacoustics.room.Room`` allows to compute the theoretical
  RT60 of the room according to Eyring or Sabine formula
- The method ``measure_rt60`` in ``pyroomacoustics.room.Room`` allows to measure the RT60 of
  the simulated RIRs

Changes in the Room Class
~~~~~~~~~~~~~~~~~~~~~~~~~

- Deep refactor of Room class. The constructor arguments have changed
- No more ``sigma2_awgn``, noise is now handled in ``pyroomacoustics.Room.simulate`` method
- The way absorption is handled has changed. The scalar variables
  ``absorption`` are deprecated in favor of ``pyroomacoustics.materials.Material``
- Complete refactor of libroom, the compiled extension module responsible for the
  room simulation, into C++. The bindings to python are now done using pybind11.
- Removes the pure Python room simulator as it was really slow
- ``pyroomacoustics.transform.analysis``, ``pyroomacoustics.transform.synthesis``,
  ``pyroomacoustics.transform.compute_synthesis_window``, have been deprecated in favor of
  ``pyroomacoustics.transform.stft.analysis``, ``pyroomacoustics.transform.stft.synthesis``,
  ``pyroomacoustics.transform.stft.compute_synthesis_window``.
- ``pyroomacoustics.Room`` has a new method ``add`` that can be used to add
  either a ``SoundSource``, or a ``MicrophoneArray`` object.  Subsequent calls
  to the method will always add source/microphones. There exists also methods
  ``add_source`` and ``add_microphone`` that can be used to add
  source/microphone via coordinates. The method ``add_microphone_array`` can be
  used to add a ``MicrophoneArray`` object, or a 2D array containing the
  locations of several microphones in its columns.  While the
  ``add_microphone_array`` method used to replace the existing array by the
  argument, the new behavior is to add in addition to other microphones already
  present.

Bugfix
~~~~~~

- From Issue #150, increase max iterations to check if point is inside room
- Issues #117 #163, adds project file `pyproject.toml` so that pip can know which dependencies are necessary for setup
- Fixed some bugs in the documentation
- Fixed normalization part in FastMNMF

Added
~~~~~~~

- Added `room_isinside_max_iter` in `parameters.py`
- Default set to 20 rather than 5 as it was in `pyroomacoustics.room.Room.isinside`
- Added Binder link in the README for online interactive demo

Changed
~~~~~~~

- Changed while loop to iterate up to `room_isinside_max_iter` in `pyroomacoustics.room.Room.isinside`
- Changed initialization of FastMNMF to accelerate convergence
- Fixed bug in doa/tops (float -> integer division)
- Added vectorised functions in MUSIC 
- Use the vectorised functions in _process of MUSIC


`0.3.1`_ - 2019-11-06
---------------------

Bugfix
~~~~~~

- Fixed a non-unicode character in ``pyroomacoustics.experimental.rt60`` breaking
  the tests

`0.3.0`_ - 2019-11-06
---------------------

Added
~~~~~

- The routine ``pyroomacoustics.experimental.measure_rt60`` to automatically
  measure the reverberation time of impulse responses. This is useful for
  measured and simulated responses.

Bugfix
~~~~~~

- Fixed docstring and an argument of `pyroomacoustics.bss.ilrma`

`0.2.0`_ - 2019-09-04
---------------------

Added
~~~~~

- Added FastMNMF (Fast Multichannel Nonnegative Matrix Factorization) to ``bss`` subpackage.
- Griffin-Lim algorithm for phase reconstruction from STFT magnitude measurements.

Changed
~~~~~~~

- Removed the supperfluous warnings in `pyroomacoustics.transform.stft`.
- Add option in `pyroomacoustics.room.Room.plot_rir` to set pair of channels
  to plot; useful when there's too many impulse responses.
- Add some window functions in `windows.py` and rearranged it in alphabetical order
- Fixed various warnings in tests.
- Faster implementation of AuxIVA that also includes OverIVA (more mics than sources).
  It also comes with a slightly changed API, Laplace and time-varying Gauss statistical
  models, and two possible initialization schemes.
- Faster implementation of ILRMA.
- SparseAuxIVA has slightly changed API, ``f_contrast`` has been replaced by ``model``
  keyword argument.

Bugfix
~~~~~~

- Set ``rcond=None`` in all calls to ``numpy.linalg.lstsq`` to remove a ``FutureWarning``
- Add a lower bound to activations in ``pyroomacoustics.bss.auxiva`` to avoid
  underflow and divide by zero.
- Fixed a memory leak in the C engine for polyhedral room (issue #116).
- Fixed problem caused by dependency of setup.py on Cython (Issue #117)

`0.1.23`_ - 2019-04-17
----------------------

Bugfix
~~~~~~

- Expose ``mu`` parameter for ``adaptive.subband_lms.SubbandLMS``.
- Add SSL context to ``download_uncompress`` and unit test; error for Python 2.7.


`0.1.22`_ - 2019-04-11
----------------------

Added
~~~~~
- Added "precision" parameter to "stft" class to choose between 'single' (float32/complex64) or 'double'
  (float64/complex128) for processing precision.
- Unified unit test file for frequency-domain souce separation methods.
- New algorithm for blind source separation (BSS): Sparse Independent Vector Analysis (SparseAuxIVA).

Changed
~~~~~~~

- Few README improvements

Bugfix
~~~~~~

- Remove ``np.squeeze`` in STFT as it caused errors when an axis that shouldn't
  be squeezed was equal to 1.
- ``Beamformer.process`` was using old (non-existent) STFT function. Changed to
  using one-shot function from ``transform`` module.
- Fixed a bug in ``utilities.fractional_delay_filter_bank`` that would cause the
  function to crash on some inputs (`issue #87 <https://github.com/LCAV/pyroomacoustics/issues/87>`__).


`0.1.21`_ - 2018-12-20
----------------------

Added
~~~~~

- Adds several options to ``pyroomacoustics.room.Room.simulate`` to finely
  control the SNR of the microphone signals and also return the microphone
  signals with individual sources, prior to mix (useful for BSS evaluation)
- Add subspace denoising approach in ``pyroomacoustics.denoise.subspace``.
- Add iterative Wiener filtering approach for single channel denoising in
  ``pyroomacoustics.denoise.iterative_wiener``.


Changed
~~~~~~~

- Add build instructions for python 3.7 and wheels for Mac OS X in the
  continuous integration (Travis and Appveyor)
- Limits imports of matplotlib to within plotting functions so that the
  matplotlib backend can still be changed, even after importing pyroomacoustics
- Better Vectorization of the computations in ``pyroomacoustics.bss.auxiva``

Bugfix
~~~~~~

- Corrects a bug that causes different behavior whether sources are provided to the constructor of ``Room`` or to the ``add_source`` method
- Corrects a typo in ``pyroomacoustics.SoundSource.add_signal``
- Corrects a bug in the update of the demixing matrix in ``pyroomacoustics.bss.auxiva``
- Corrects invalid memory access in the ``pyroomacoustics.build_rir`` cython accelerator
  and adds a unit test that checks the cython code output is correct
- Fix bad handling of 1D `b` vectors in ```pyroomacoustics.levinson``.

`0.1.20`_ - 2018-10-04
----------------------

Added
~~~~~

- STFT tutorial and demo notebook.
- New algorithm for blind source separation (BSS): Independent Low-Rank Matrix Analysis (ILRMA)

Changed
~~~~~~~

- Matplotlib is not a hard requirement anymore. When matplotlib is not
  installed, only a warning is issued on plotting commands. This is useful
  to run pyroomacoustics on headless servers that might not have matplotlib
  installed
- Removed dependencies on ``joblib`` and ``requests`` packages
- Apply ``matplotlib.pyplot.tight_layout`` in ``pyroomacoustics.Room.plot_rir``

Bugfix
~~~~~~

- Monaural signals are now properly handled in one-shot stft/istft
- Corrected check of size of absorption coefficients list in ``Room.from_corners``

`0.1.19`_ - 2018-09-24
----------------------

Added
~~~~~

- Added noise reduction sub-package ``denoise`` with spectral subtraction
  class and example.
- Renamed ``realtime`` to ``transform`` and added deprecation warning.
- Added a cython function to efficiently compute the fractional delays in the room
  impulse response from time delays and attenuations
- `notebooks` folder.
- Demo IPython notebook (with WAV files) of several features of the package.
- Wrapper for Google's Speech Command Dataset and an example usage script in ``examples``.
- Lots of new features in the ``pyroomacoustics.realtime`` subpackage

  * The ``STFT`` class can now be used both for frame-by-frame processing
    or for bulk processing
  * The functionality will replace the methods ``pyroomacoustics.stft``,
    ``pyroomacoustics.istft``, ``pyroomacoustics.overlap_add``, etc,
  * The **new** function ``pyroomacoustics.realtime.compute_synthesis_window``
    computes the optimal synthesis window given an analysis window and
    the frame shift
  * Extensive tests for the ``pyroomacoustics.realtime`` module
  * Convenience functions ``pyroomacoustics.realtime.analysis`` and
    ``pyroomacoustics.realtime.synthesis`` with an interface similar
    to ``pyroomacoustics.stft`` and ``pyroomacoustics.istft`` (which
    are now deprecated and will disappear soon)
  * The ordering of axis in the output from bulk STFT is now
    ``(n_frames, n_frequencies, n_channels)``
  * Support for Intel's ``mkl_fft`` `package <https://github.com/IntelPython/mkl_fft>`_
  * ``axis`` (along which to perform DFT) and ``bits`` parameters for ``DFT`` class.

Changed
~~~~~~~

- Improved documentation and docstrings
- Using now the built-in RIR generator in `examples/doa_algorithms.py`
- Improved the download/uncompress function for large datasets
- Dusted the code for plotting on the sphere in ``pyroomacoustics.doa.grid.GridSphere``

Deprecation Notice
~~~~~~~~~~~~~~~~~~

- The methods ``pyroomacoustics.stft``, ``pyroomacoustics.istft``,
  ``pyroomacoustics.overlap_add``, etc, are now **deprecated**
  and will be removed in the near future

`0.1.18`_ - 2018-04-24
----------------------

Added
~~~~~

- Added AuxIVA (independent vector analysis) to ``bss`` subpackage.
- Added BSS IVA example

Changed
~~~~~~~

- Moved Trinicon blind source separation algorithm to ``bss`` subpackage.

Bugfix
~~~~~~

- Correct a bug that causes 1st order sources to be generated for `max_order==0`
  in pure python code

`0.1.17`_ - 2018-03-23
----------------------

Bugfix
~~~~~~

- Fixed issue #22 on github. Added INCREF before returning Py_None in C extension.

`0.1.16`_ - 2018-03-06
----------------------

Added
~~~~~

- Base classes for Dataset and Sample in ``pyroomacoustics.datasets``
- Methods to filter datasets according to the metadata of samples
- Deprecation warning for the TimitCorpus interface

Changed
~~~~~~~

- Add list of speakers and sentences from CMU ARCTIC
- CMUArcticDatabase basedir is now the top directory where CMU_ARCTIC database
  should be saved. Not the directory above as it previously was.
- Libroom C extension is now a proper package. It can be imported.
- Libroom C extension now compiles on windows with python>=3.5.


`0.1.15`_ - 2018-02-23
----------------------

Bugfix
~~~~~~

- Added ``pyroomacoustics.datasets`` to list of sub-packages in ``setup.py``


`0.1.14`_ - 2018-02-20
----------------------

Added
~~~~~

-  Changelog
-  CMU ARCTIC corpus wrapper in ``pyroomacoustics.datasets``

Changed
~~~~~~~

-  Moved TIMIT corpus wrapper from ``pyroomacoustics.recognition`` module to sub-package
   ``pyroomacoustics.datasets.timit``


.. _Unreleased: https://github.com/LCAV/pyroomacoustics/compare/v0.7.4...master
.. _0.7.4: https://github.com/LCAV/pyroomacoustics/compare/v0.7.3...v0.7.4
.. _0.7.3: https://github.com/LCAV/pyroomacoustics/compare/v0.7.2...v0.7.3
.. _0.7.2: https://github.com/LCAV/pyroomacoustics/compare/v0.7.1...v0.7.2
.. _0.7.1: https://github.com/LCAV/pyroomacoustics/compare/v0.7.0...v0.7.1
.. _0.7.0: https://github.com/LCAV/pyroomacoustics/compare/v0.6.0...v0.7.0
.. _0.6.0: https://github.com/LCAV/pyroomacoustics/compare/v0.5.0...v0.6.0
.. _0.5.0: https://github.com/LCAV/pyroomacoustics/compare/v0.4.3...v0.5.0
.. _0.4.3: https://github.com/LCAV/pyroomacoustics/compare/v0.4.2...v0.4.3
.. _0.4.2: https://github.com/LCAV/pyroomacoustics/compare/v0.4.1...v0.4.2
.. _0.4.1: https://github.com/LCAV/pyroomacoustics/compare/v0.4.0...v0.4.1
.. _0.4.0: https://github.com/LCAV/pyroomacoustics/compare/v0.3.1...v0.4.0
.. _0.3.1: https://github.com/LCAV/pyroomacoustics/compare/v0.3.0...v0.3.1
.. _0.3.0: https://github.com/LCAV/pyroomacoustics/compare/v0.2.0...v0.3.0
.. _0.2.0: https://github.com/LCAV/pyroomacoustics/compare/v0.1.23...v0.2.0
.. _0.1.23: https://github.com/LCAV/pyroomacoustics/compare/v0.1.22...v0.1.23
.. _0.1.22: https://github.com/LCAV/pyroomacoustics/compare/v0.1.21...v0.1.22
.. _0.1.21: https://github.com/LCAV/pyroomacoustics/compare/v0.1.20...v0.1.21
.. _0.1.20: https://github.com/LCAV/pyroomacoustics/compare/v0.1.19...v0.1.20
.. _0.1.19: https://github.com/LCAV/pyroomacoustics/compare/v0.1.18...v0.1.19
.. _0.1.18: https://github.com/LCAV/pyroomacoustics/compare/v0.1.17...v0.1.18
.. _0.1.17: https://github.com/LCAV/pyroomacoustics/compare/v0.1.16...v0.1.17
.. _0.1.16: https://github.com/LCAV/pyroomacoustics/compare/v0.1.15...v0.1.16
.. _0.1.15: https://github.com/LCAV/pyroomacoustics/compare/v0.1.14...v0.1.15
.. _0.1.14: https://github.com/LCAV/pyroomacoustics/compare/v0.1.13...v0.1.14
