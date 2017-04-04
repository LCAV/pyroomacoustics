#!/usr/bin/env python

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

libroom_ext = Extension('libroom',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '1')],
                    extra_compile_args = ['-Wall', '-O3'],
                    sources = [src_dir + f for f in files])

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
        name='pyroomacoustics',

        version='1.0',

        description='A simple framework for room acoustics and signal processing in Python.',
        long_description=long_description,

        author='Laboratory for Audiovisual Communications, EPFL',

        author_email='fakufaku@gmail.ch',

        url='https://github.com/LCAV/pyroomacoustics',

        license='GPL-3.0',

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
            'License :: OSI Approved :: GPL-3.0 License',

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
