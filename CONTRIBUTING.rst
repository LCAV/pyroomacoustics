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
   ``conda create -n mkdocs27 python=2.7 sphinx numpydoc mock sphinx-rtd-theme``
-  Switch to the environment: ``source activate mkdocs27``
-  Navigate to the ``docs`` folder and run: ``./make_apidoc.sh``
-  Build and view the documentation locally with: ``make html``
-  Open in your browser: ``docs/_build/html/index.html``

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
