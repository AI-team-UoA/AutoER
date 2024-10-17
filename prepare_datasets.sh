#!/bin/bash

echo "Downloading datasets..."
curl -L -o autoer_data.tar.gz https://zenodo.org/records/13946189/files/autoer_data.tar.gz
curl -L -o zeroer_like_datasets.tar.gz https://zenodo.org/records/13946189/files/zeroer_like_datasets.tar.gz

echo "Extracting datasets... in specific folder"

tar -xf autoer_data.tar.gz
tar -xf zeroer_like_datasets.tar.gz

rm autoer_data.tar.gz
rm zeroer_like_datasets.tar.gz

echo "Done!"