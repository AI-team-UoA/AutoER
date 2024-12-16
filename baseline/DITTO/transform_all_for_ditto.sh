#!/bin/bash

# Define the base path for the JSON files
CONFIG_DIR="../../data/configs"

# Loop through D1 to D10 and run the Python script
for i in {1..10}; do
    DATAJSON="${CONFIG_DIR}/D${i}.json"
    echo "Running blocking.py for ${DATAJSON}..."
    python blocking.py --datajson "${DATAJSON}"
done

echo "All datasets processed."
