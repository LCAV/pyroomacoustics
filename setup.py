#!/usr/bin/env python
from __future__ import print_function

from pyroomacoustics import __version__

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    print("import error")
    from distutils.core import setup
    from distutils.extension import Extension

# To use a consistent encoding
from codecs import open
from os import path

# build C extension for image source model
src_dir = 'pyroomacoustics/c_package/'
files = ['wall.c', 'linalg.c', 'room.c', 'is_list.c', 'shoebox.c']

libroom_ext = Extension('pyroomacoustics.c_package.libroom',
                    extra_compile_args = ['-Wall', '-O3', '-std=c99'],
                    sources = [src_dir + f for f in files],
                    include_dirs=[src_dir])

libroom_lib = ('pyroomacoustics.c_package.libroom', {
    'sources': [src_dir + f for f in files],
    'extra_compile_args': ['-Wall', '-O3', '-std=c99'],
    'include_dirs':[src_dir]
    })

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup_kwargs = dict(
        name='pyroomacoustics',

        version=__version__,

        description='A simple framework for room acoustics and signal processing in Python.',
        long_description=long_description,

        author='Laboratory for Audiovisual Communications, EPFL',

        author_email='fakufaku@gmail.ch',

        url='https://github.com/LCAV/pyroomacoustics',

        license='GPL3',

        # You can just specify the packages manually here if your project is
        # simple. Or you can use find_packages().
        packages=['pyroomacoustics', 'pyroomacoustics.c_package'],

        # Libroom C extension
        ext_modules=[libroom_ext],

        install_requires=[
            'numpy',
            'scipy',
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
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Build Tools',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

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
        keywords='room acoustics signal processing',
)

try:
    # Try to build everything first
    setup(**setup_kwargs)

except:
    # Retry without the C module
    print("Error. Probably building C extension failed. Retrying without.")
    setup_kwargs.pop('ext_modules')
    setup(**setup_kwargs)
    

