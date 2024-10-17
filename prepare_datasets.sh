#!/bin/bash

echo "Downloading datasets..."
curl -L -o autoer_data.tar.gz https://zenodo.org/records/13946189/files/autoer_data.tar.gz
curl -L -o zeroer_like_datasets.tar.gz https://zenodo.org/records/13946189/files/zeroer_like_datasets.tar.gz

echo "Extracting datasets... in specific folder"

mkdir -p ./data
tar -xzf autoer_data.tar.gz -C data

mkdir -p ./baselines/zeroer/datasets
tar -xzf zeroer_like_datasets.tar.gz -C baselines/zeroer/datasets

echo "Done!"