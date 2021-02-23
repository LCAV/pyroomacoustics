#!/bin/bash
set -e -x

# Install a system package required by our library
sudo yum install -y atlas-devel

# Compile wheels
for PYBIN in /opt/python/*cp38/bin; do
    # sudo ${PYBIN}/pip install -r ./dev-requirements.txt
    sudo ${PYBIN}/pip install -U pip setuptools
    sudo ${PYBIN}/pip install wheel numpy Cython pybind11
    #LDSHARED="$CC -shared"  \
        #${PYBIN}/pip wheel . -w wheelhouse/ \
        #--no-deps \
        #--build-option --plat-name=manylinux2014_aarch64

    LDSHARED="$CC -shared" ${PYBIN}/python ./setup.py bdist_wheel -b wheelhouse/ \
        --plat-name manylinux2014_aarch64
done

# Bundle external shared libraries into the wheels
#for whl in wheelhouse/*.whl; do
    #auditwheel repair $whl -w ./wheelhouse/
#done

# Install packages and test
#for PYBIN in /opt/python/*/bin/; do
    #sudo ${PYBIN}/pip install python-manylinux-demo --no-index -f ./wheelhouse
    #(cd $HOME; ${PYBIN}/nosetests pymanylinuxdemo)
#done
