#!/bin/bash

# Function to kill processes based on user confirmation
kill_processes_with_confirmation() {
    ps aux | grep 'python -u regression_with_automl.py' | grep -v grep | while read -r line; do
        pid=$(echo $line | awk '{print $2}')
        cmd=$(echo $line | awk '{print $11, $12, $13}')
        echo "Found process: $cmd (PID: $pid)"
        read -p "Do you want to kill this process? (y/n): " choice
        if [ "$choice" = "y" ]; then
            cmd=$(ps -p $pid -o cmd=)
            echo "Found process: $cmd (PID: $pid)"
            echo "Killing process ID $pid"
            kill -9 $pid
        else
            echo "Skipping process ID $pid"
        fi
    done
}

# Function to forcefully kill all processes without confirmation
force_kill_all_processes() {
    ps aux | grep 'python -u regression_with_automl.py' | grep -v grep | awk '{print $2}' | while read pid; do
        # print all command
        cmd=$(ps -p $pid -o cmd=)
        echo "Found process: $cmd (PID: $pid)"
        echo "Killing process ID $pid"
        kill -9 $pid
    done
    echo "All matching processes have been terminated."
}

# Check for the force flag
if [ "$1" = "-f" ]; then
    force_kill_all_processes
else
    kill_processes_with_confirmation
fi
