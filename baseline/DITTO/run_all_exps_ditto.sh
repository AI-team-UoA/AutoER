#!/bin/bash

# Check if container ID is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <container_id_or_name>"
  exit 1
fi

CONTAINER_ID=$1

# Array of tasks D1 to D10
tasks=(D1 D2 D3 D4 D5 D6 D7 D8 D9 D10)

# Loop through each task and execute the training command inside the container
for task in "${tasks[@]}"; do
  echo "Running task $task..."
  docker exec -it $CONTAINER_ID bash -c "cd /workspace/ditto && CUDA_VISIBLE_DEVICES=0 python /workspace/ditto/train_ditto.py \
    --task AutoER/$task  \
    --batch_size 16 \
    --max_len 256 \
    --lr 3e-5 \
    --n_epochs 5 \
    --lm roberta \
    --fp16 \
    --da del \
    --dk product \
    --summarize > /workspace/ditto/logs/$task.log 2>&1"
  echo "Task $task completed. Log saved to /workspace/ditto/logs/$task.log"
done

# Optionally, copy logs from the container to the host machine
read -p "Do you want to copy log files from the container to the host? (y/n): " copy_logs
if [ "$copy_logs" == "y" ]; then
  for task in "${tasks[@]}"; do
    docker cp $CONTAINER_ID:/workspace/ditto/logs/$task.log ./logs
    echo "Copied $task.log from container to the current directory."
  done
fi