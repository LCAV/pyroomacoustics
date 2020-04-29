Contributing
============

If you want to contribute to ``pyroomacoustics`` and make it better,
your help is very welcome. Contributing is also a great way to learn
more about the package itself.

Ways to contribute
~~~~~~~~~~~~~~~~~~

-  File bug reports
-  Improvements to the documentation are always more than welcome.
   Keeping a good clean documentation is a challenging task and any help
   is appreciated.
-  Feature requests
-  If you implemented an extra DOA/adaptive filter/beamforming
   algorithm: that's awesome! We'd love to add it to the package.
-  Suggestion of improvements to the code base are also welcome.

Coding style
~~~~~~~~~~~~

We try to stick to `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__
as much as possible. Variables, functions, modules and packages should
be in lowercase with underscores. Class names in CamelCase.

Documentation
~~~~~~~~~~~~~

Docstrings should follow the `numpydoc
style <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__.

We recommend the following steps for generating the documentation:

-  Create a separate environment, e.g. with Anaconda, as such:
   ``conda create -n mkdocs37 python=3.7 sphinx numpydoc mock sphinx_rtd_theme``
-  Switch to the environment: ``source activate mkdocs37``
-  Navigate to the ``docs`` folder and run: ``./make_apidoc.sh``
-  Build and view the documentation locally with: ``make html``
-  Open in your browser: ``docs/_build/html/index.html``

Develop Locally
~~~~~~~~~~~~~~~

It can be convenient to develop and run tests locally.  In contrast to only
using the package, you will then also need to compile the C++ extension for
that. On Mac and Linux, GCC is required, while Visual C++ 14.0 is necessary for
`windows <https://wiki.python.org/moin/WindowsCompilers>`__. 

1. Get the source code. Use recursive close so that Eigen (a sub-module of this
   repository) is also downloaded.

   .. code-block:: shell

       git clone --recursive git@github.com:LCAV/pyroomacoustics.git

   Alternatively, you can clone without the `--recursive` flag and directly
   install the Eigen library. For macOS, you can find installation instruction
   here: https://stackoverflow.com/a/35658421. After installation you can
   create a symbolic link as such:

    .. code-block:: shell

        ln -s PATH_TO_EIGEN pyroomacoustics/libroom_src/ext/eigen/Eigen

2. Install a few pre-requisites

    .. code-block:: shell

        pip install numpy Cython pybind11

3. Compile locally

   .. code-block:: shell

         python setup.py build_ext --inplace

   On recent Mac OS (Mojave), it is necessary in some cases to add a
   higher deployment target

   .. code-block:: shell

         MACOSX_DEPLOYMENT_TARGET=10.9 python setup.py build_ext --inplace

4. Update ``$PYTHONPATH`` so that python knows where to find the local package

   .. code-block:: shell

      # Linux/Mac
      export PYTHONPATH=<path_to_pyroomacoustics>:$PYTHONPATH

   For windows, see `this question <https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows>`__
   on stackoverflow.

5. Install the dependencies listed in ``requirements.txt``

   .. code-block:: shell

      pip install -r requirements.txt

6. Now fire up ``python`` or ``ipython`` and check that the package can be
   imported

   .. code-block:: python

      import pyroomacoustics as pra

Unit Tests
~~~~~~~~~~

As much as possible, for every new function added to the code base, add
a short test script in ``pyroomacoustics/tests``. The names of the
script and the functions running the test should be prefixed by
``test_``. The tests are started by running ``nosetests`` at the root of
the package.

How to make a clean pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Look for a project's contribution instructions. If there are any, follow
them.

-  Create a personal fork of the project on Github.
-  Clone the fork on your local machine. Your remote repo on Github is
   called ``origin``.
-  Add the original repository as a remote called ``upstream``.
-  If you created your fork a while ago be sure to pull upstream changes
   into your local repository.
-  Create a new branch to work on! Branch from ``develop`` if it exists,
   else from ``master``.
-  Implement/fix your feature, comment your code.
-  Follow the code style of the project, including indentation.
-  If the project has tests run them!
-  Write or adapt tests as needed.
-  Add or change the documentation as needed.
-  Squash your commits into a single commit with git's `interactive
   rebase <https://help.github.com/articles/interactive-rebase>`__.
   Create a new branch if necessary.
-  Push your branch to your fork on Github, the remote ``origin``.
-  From your fork open a pull request in the correct branch. Target the
   project's ``develop`` branch if there is one, else go for ``master``!
-  …
-  If the maintainer requests further changes just push them to your
   branch. The PR will be updated automatically.
-  Once the pull request is approved and merged you can pull the changes
   from ``upstream`` to your local repo and delete your extra
   branch(es).

And last but not least: Always write your commit messages in the present
tense. Your commit message should describe what the commit, when
applied, does to the code – not what you did to the code.

Reference
---------

This guide is based on the nice template by
`@MarcDiethelm <https://github.com/MarcDiethelm/contributing>`__ available
under MIT License.
