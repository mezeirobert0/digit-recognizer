#!/usr/bin/bash

unzip datasets.zip -d ./datasets
python -m venv ./src/.venv
source ./src/.venv/bin/activate
pip install numpy pandas 
cd ./src
python network.py > ../python_output
cd ..
deactivate
git add .
git commit -m "Train neural network on MNIST dataset, new weights and biases added to a folder"
git push origin main
# screen -X -S "training_model" quit
