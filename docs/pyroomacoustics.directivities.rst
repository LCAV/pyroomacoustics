Directional Sources and Microphones
===================================

.. automodule:: pyroomacoustics.directivities
   :members:
   :undoc-members:
   :show-inheritance:

Analytic Directional Responses
------------------------------

.. automodule:: pyroomacoustics.directivities.analytic
   :members:
   :undoc-members:
   :show-inheritance:

Real Spherical Harmonics Directivities
--------------------------------------

.. automodule:: pyroomacoustics.directivities.harmonics
   :members:
   :undoc-members:
   :show-inheritance:

Measured Directivities
----------------------

.. automodule:: pyroomacoustics.directivities.measured
   :members:
   :undoc-members:
   :show-inheritance:

Built-in SOFA Files Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyroomacoustics.datasets.sofa
   :members: SOFADatabase, get_sofa_db
   :noindex:


Reading Other or Custom File Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to read other file types by providing a custom reader function to
:py:class:`~pyroomacoustics.directivities.measured.MeasuredDirectivityFile` with the
argument ``file_reader_callback``.
The function should have the same signature as :py:func:`~pyroomacoustics.directivities.sofa.open_sofa_file`.


SOFA File Readers
.................

.. automodule:: pyroomacoustics.directivities.sofa
   :members:
   :show-inheritance:

Direction of the Patterns
-------------------------

.. automodule:: pyroomacoustics.directivities.direction
   :members:
   :show-inheritance:


Creating New Types of Directivities
-----------------------------------

.. automodule:: pyroomacoustics.directivities.base
   :members:
   :undoc-members:
   :show-inheritance:

Spherical Interpolation
-----------------------

.. automodule:: pyroomacoustics.directivities.interp
   :members: spherical_interpolation

Numerical Spherical Integral
----------------------------

.. automodule:: pyroomacoustics.directivities.integration
   :members: spherical_integral
