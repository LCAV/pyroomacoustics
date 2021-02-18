DOCKER_IMAGE=quay.io/pypa/manylinux2014_aarch64
PLAT=manylinux2014_aarch64
docker run --rm -e PLAT=$PLAT -v `pwd`:/io $DOCKER_IMAGE $PRE_CMD /io/travis/build-wheels.sh
