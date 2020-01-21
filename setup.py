#!/usr/bin/env python
from __future__ import print_function
import numpy
import os, sys

# import version from file
with open('pyroomacoustics/version.py') as f:
    exec(f.read())

try:
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    from setuptools import distutils
except ImportError:
    print("Setuptools unavailable. Falling back to distutils.")
    import distutils
    from distutils.core import setup
    from distutils.extension import Extension
    from distutils.command.build_ext import build_ext

# To use a consistent encoding
from codecs import open
from os import path


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


# build C extension for image source model
libroom_src_dir = 'pyroomacoustics/libroom_src'
libroom_files = [ os.path.join(libroom_src_dir, f)
                for f in
                [
                    'room.hpp', 'room.cpp',
                    'wall.hpp', 'wall.cpp',
                    'microphone.hpp',
                    'geometry.hpp', 'geometry.cpp',
                    'common.hpp',
                    'libroom.cpp',
                    ]
                ]
ext_modules = [
        Extension(
            'pyroomacoustics.libroom',
            [ os.path.join(libroom_src_dir, f)
                for f in ['libroom.cpp'] ],
            depends=libroom_files,
            include_dirs=[
                '.',
                libroom_src_dir,
                str(get_pybind_include()),
                str(get_pybind_include(user=True)),
                os.path.join(libroom_src_dir, 'ext/eigen'),
                ],
            language='c++',
            extra_compile_args = ['-DEIGEN_MPL2_ONLY', '-Wall', '-O3', '-DEIGEN_NO_DEBUG'],
            ),
        Extension(
            'pyroomacoustics.build_rir',
            ["pyroomacoustics/build_rir.pyx"],
            language='c',
            extra_compile_args = [],
            ),
        ]

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


### Build Tools (taken from pybind11 example) ###

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            if ext.language == 'c++':
                ext.extra_compile_args += opts
                ext.extra_link_args += opts
        build_ext.build_extensions(self)

### Build Tools End ###


setup_kwargs = dict(
        name='pyroomacoustics',

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
            'pyroomacoustics.doa', 
            'pyroomacoustics.adaptive',
            'pyroomacoustics.transform',
            'pyroomacoustics.experimental',
            'pyroomacoustics.datasets',
            'pyroomacoustics.bss',
            'pyroomacoustics.denoise',
            'pyroomacoustics.phase',
            ],

        # Libroom C extension
        ext_modules=ext_modules,

        # Necessary to keep the source files
        package_data={
            'pyroomacoustics': ['*.pxd', '*.pyx',],
            },

        install_requires=[
            'Cython',
            'numpy',
            'scipy>=0.18.0',
            'pybind11>=2.2',
            ],

        cmdclass={'build_ext': BuildExt},  # taken from pybind11 example
        zip_safe=False,

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

setup(**setup_kwargs)
