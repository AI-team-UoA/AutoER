#!/bin/bash

D="census"

# python create_test_trials.csv --data $D

# # Wait for the first script to finish
# wait

# Run the first Python script
python evaluate.py --data $D

# Wait for the first script to finish
# wait

# Run the second Python script
# python performance_per_census_dataset.py
