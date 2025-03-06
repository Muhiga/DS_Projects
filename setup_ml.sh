#!/bin/bash

#set up python virtual environment
python3 -m venv venv

#activate the virtual environment
source venv/bin/activate

#install the necessary libraries
pip install numpy pandas seaborn matplotlib scikit-learn xgboost

