#!/bin/bash

set -e

apt-get update && apt-get upgrade -y

apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    wget \
    vim \
    openssh-server \
    libgl1-mesa-glx \
    libglib2.0-0

add-apt-repository ppa:deadsnakes/ppa -y
apt-get update && apt-get install -y --no-install-recommends python3.10 python3.10-dev python3.10-distutils
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
