#!/bin/bash

python3 -m venv venv
pip3 install torch psutil numpy tqdmnumpy tqdmnumpy tqdmnumpy tqdm

source venv/bin/activate

python test.py

deactivate
