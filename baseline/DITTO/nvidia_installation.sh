#!/bin/bash

# Stop on errors
set -e

echo "Step 1: Detecting OS distribution..."
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
echo "OS detected: $distribution"

echo "Step 2: Cleaning up old NVIDIA repository configurations..."
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo rm -f /etc/apt/sources.list.d/nvidia-docker.list

echo "Step 3: Adding NVIDIA GPG key..."
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor | sudo tee /usr/share/keyrings/nvidia-container-keyring.gpg > /dev/null

echo "Step 4: Adding NVIDIA repository..."
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-keyring.gpg] https://#' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "Step 5: Updating package list..."
sudo apt-get update

echo "Step 6: Installing NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

echo "Step 7: Restarting Docker service..."
sudo systemctl restart docker

echo "Step 8: Verifying NVIDIA setup..."
if sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi; then
    echo "NVIDIA Container Toolkit installed successfully and GPU is accessible!"
else
    echo "GPU verification failed. Please check your GPU drivers and setup."
    exit 1
fi


# CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task Structured/Amazon-Google --batch_size 32 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize
