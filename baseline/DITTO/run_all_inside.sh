#!/bin/bash

# Array of tasks D1 to D10
# tasks=(D1 D2 D3 D4 D5 D6 D7 D8 D9 D10)
tasks=(D6 D7 D10)

# Loop through each task and execute the training command inside the container
for task in "${tasks[@]}"; do
  {
    echo "Running task $task..."
    cd /workspace/ditto && CUDA_VISIBLE_DEVICES=0 python -u train_ditto.py \
      --task AutoER/$task  \
      --batch_size 16 \
      --max_len 256 \
      --lr 3e-5 \
      --n_epochs 5 \
      --lm roberta \
      --fp16 \
      --da del \
      --dk product \
      # --summarize
    echo "Task $task completed. Log saved to /workspace/ditto/logs/$task.log"
  } >> /workspace/ditto/logs/$task.log 2>&1

done

# # Optionally, copy logs from container directory to a host machine directory
# read -p "Do you want to copy log files to the host? (y/n): " copy_logs
# if [ "$copy_logs" == "y" ]; then
#   for task in "${tasks[@]}"; do
#     cp /workspace/ditto/$task.log ./
#     echo "Copied $task.log to the current directory."
#   done
# fi
