#!/usr/bin/bash

unzip datasets.zip -d ./datasets
python -m venv ./src/.venv
source ./src/.venv/bin/activate
pip install -r requirements.txt
cd ./src
python Network.py
cd ..
deactivate
git add .
git commit -m "Train neural network on MNIST dataset, with the new weights and biases being added in a folder"
git push -u origin main
