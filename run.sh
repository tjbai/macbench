#!/bin/bash

python3 -m venv venv

source venv/bin/activate
pip3 install torch psutil numpy tqdm
python test.py

deactivate
