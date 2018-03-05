#!/usr/bin/env python
from __future__ import print_function
import numpy

# import version from file
with open('pyroomacoustics/version.py') as f:
    exec(f.read())

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    print("Setuptools unavailable. Falling back to distutils.")
    from distutils.core import setup
    from distutils.extension import Extension

# To use a consistent encoding
from codecs import open
from os import path

# build C extension for image source model
src_dir = 'pyroomacoustics/c_package'
files = ['libroom.c', 'wall.c', 'linalg.c', 'room.c', 'is_list.c', 'shoebox.c']

libroom_ext = Extension('pyroomacoustics.c_package.libroom',
                    extra_compile_args = ['-Wall', '-O3', '-std=c99'],
                    sources = [src_dir + '/' + f for f in files],
                    include_dirs=[src_dir,numpy.get_include()])

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup_kwargs = dict(
        name='pyroomacoustics',

        version=__version__,

        description='A simple framework for room acoustics and audio processing in Python.',
        long_description=long_description,

        author='Laboratory for Audiovisual Communications, EPFL',

        author_email='fakufaku@gmail.ch',

        url='https://github.com/LCAV/pyroomacoustics',

        license='MIT',

        # You can just specify the packages manually here if your project is
        # simple. Or you can use find_packages().
        packages=[
            'pyroomacoustics', 
            'pyroomacoustics.c_package', 
            'pyroomacoustics.doa', 
            'pyroomacoustics.adaptive',
            'pyroomacoustics.realtime',
            'pyroomacoustics.experimental',
            'pyroomacoustics.datasets',
            ],

        # Libroom C extension
        ext_modules=[libroom_ext],

        install_requires=[
            'numpy',
            'scipy>=0.18.0',
            'matplotlib',
            'joblib',
            ],

        test_suite='nose.collector',
        tests_require=['nose'],

        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 4 - Beta',

            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Intended Audience :: Information Technology',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Multimedia :: Sound/Audio :: Speech',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: MIT License',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            #'Programming Language :: Python :: 3.3',
            #'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            ],

        # What does your project relate to?
        keywords='room acoustics signal processing doa beamforming adaptive',
)

try:
    # Try to build everything first
    setup(**setup_kwargs)
except:
    # Retry without the C module
    print("Error. Probably building C extension failed. Installing pure python.")
    setup_kwargs.pop('ext_modules')
    setup(**setup_kwargs)
    

