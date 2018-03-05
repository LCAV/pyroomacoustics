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

- Base classes for Dataset and Sample in ``pyroomacoustics.datasets``
- Methods to filter datasets according to the metadata of samples
- Deprecation warning for the TimitCorpus interface

Changed
~~~~~~~

- Add list of speakers and sentences from CMU ARCTIC
- CMUArcticDatabase basedir is now the top directory where CMU_ARCTIC database
  should be saved. Not the directory above as it previously was.
- :py:obj:`pyroomacoustics.c_package.libroom` is now a proper package.
  It can be imported.
- Libroom C extension now compiles on windows with python>=3.5.


`0.1.15`_ - 2018-06-23
----------------------

Bugfix
~~~~~~

- Added ``pyroomacoustics.datasets`` to list of sub-packages in ``setup.py``


`0.1.14`_ - 2018-06-20
----------------------

Added
~~~~~

-  Changelog
-  CMU ARCTIC corpus wrapper in ``pyroomacoustics.datasets``

Changed
~~~~~~~

-  Moved TIMIT corpus wrapper from ``pyroomacoustics.recognition`` module to sub-package
   ``pyroomacoustics.datasets.timit``


.. _Unreleased: https://github.com/LCAV/pyroomacoustics/compare/v0.1.15...HEAD
.. _0.1.15: https://github.com/LCAV/pyroomacoustics/compare/v0.1.14...v0.1.15
.. _0.1.14: https://github.com/LCAV/pyroomacoustics/compare/v0.1.13...v0.1.14
