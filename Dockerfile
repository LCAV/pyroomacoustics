# NB: You will have to increase the memory of your Docker containers to 4 GB.
# Less may also be sufficient, but this is necessary for building the C++ code
# for pyroomacoustics. How to increase memory: https://docs.docker.com/docker-for-mac/#resources
#
# Then you can build with: `docker build -t pyroom_container .`
# And enter container with: `docker run -it pyroom_container:latest /bin/bash`
FROM ubuntu:18.04
RUN dpkg --add-architecture i386
RUN apt-get update
RUN apt-get install -y python3-dev python3-pip
# We must reinstall pip from pypi to make sure that the dependencies are installed
# See https://github.com/pypa/pip/issues/7874
RUN python3 -m pip install --user --ignore-installed pip
RUN pip3 install pyroomacoustics
