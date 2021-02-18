#!/bin/bash
export DOCKER_IMAGE=dockcross/manylinux2014-aarch64
docker pull $DOCKER_IMAGE
docker run --rm $DOCKER_IMAGE > ./dockcross
chmod +x ./dockcross
./dockcross scripts/build-wheels.sh
ls wheelhouse/
