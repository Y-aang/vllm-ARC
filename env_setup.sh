#!/bin/bash

# Only for Reference
set -e

sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices

sudo apt install -y nvidia-cuda-toolkit

sudo apt install -y python3.11
sudo apt install -y python3-pip

sudo apt install -y python3-pip
sudo apt install -y python3.10-venv
python3 -m venv yang
source yang/bin/activate

mkdir workplace
cd workplace
git clone https://github.com/Y-aang/vllm-test-scripts.git
git clone https://github.com/Y-aang/vllm.git

cd vllm
git checkout origin/arc
git checkout -b arc
VLLM_USE_PRECOMPILED=1 pip install --editable .
